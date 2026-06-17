"""Turn a stream of per-frame piece detections into a recorded chess game.

detect_pieces.py produces, for every frame, a mapping of occupied squares to
the detected piece on them. Those detections flicker (false positives/negatives,
hands passing over the board), so we cannot treat every frame as a position.

GameTracker bridges that gap:

1. **Debounce** — a board state is only considered real once it has been seen
   unchanged for `stability_frames` consecutive frames.
2. **Snap to a legal move** — when a new stable state differs from the current
   position, we ask python-chess for the legal move whose resulting placement
   best matches the detection. Scoring the *whole* resulting position means
   castling, en passant, promotion and captures all fall out for free, and
   nonsense detections that match no legal move are simply ignored.
3. **Record** — after each accepted move the game is rewritten to a PGN file and
   a human-readable FEN log, so both files always reflect the current game even
   if the process is killed mid-match.
"""

from datetime import date
from pathlib import Path

import chess
import chess.pgn

# Piece type -> python-chess symbol (lowercase). Color decides the case.
_TYPE_SYMBOL = {
    "pawn": "p",
    "knight": "n",
    "bishop": "b",
    "rook": "r",
    "queen": "q",
    "king": "k",
}


def label_to_symbol(label: str) -> str | None:
    """Map a model class name like 'white-knight' to a python-chess symbol.

    Returns 'N' for white knight, 'n' for black knight, or None if the label
    isn't a recognized '<color>-<type>' piece name. Derived from the label text
    rather than the class index, so it survives class reordering."""
    parts = label.lower().split("-")
    if len(parts) != 2:
        return None
    color, kind = parts
    symbol = _TYPE_SYMBOL.get(kind)
    if symbol is None:
        return None
    return symbol.upper() if color == "white" else symbol


def normalize_fen(text: str) -> str:
    """Validate a user-supplied start FEN and return it in full, canonical form.

    Accepts either a complete FEN or just the piece-placement field (in which
    case White-to-move with no castling rights is assumed). Raises ValueError
    with a readable message if the position can't be parsed or isn't legal —
    recording can only track legal moves, so an illegal start is rejected."""
    text = text.strip()
    if len(text.split()) == 1:  # placement only — assume a fresh White move
        text = f"{text} w - - 0 1"
    board = chess.Board(text)  # raises ValueError on malformed FEN
    if not board.is_valid():
        raise ValueError(f"not a legal position ({board.status()!r})")
    return board.fen()


class GameTracker:
    def __init__(
        self,
        pgn_path,
        fen_path,
        *,
        start_fen: str | None = None,
        stability_frames: int = 6,
        min_match: int = 48,
        diff_tolerance: int = 1,
        debug: bool = False,
        log=print,
    ):
        self.board = chess.Board(start_fen) if start_fen else chess.Board()
        self._start_fen = self.board.fen()
        self.pgn_path = Path(pgn_path)
        self.fen_path = Path(fen_path)
        self.stability_frames = stability_frames
        # A stable state must match at least this many of the 64 squares of some
        # legal position before we trust it. Filters out garbage detections.
        self.min_match = min_match
        # "diff" mode: how many extra (spurious) changed squares to tolerate
        # when matching the observed change set to a legal move's own squares.
        self.diff_tolerance = diff_tolerance
        self.debug = debug
        self._log = log
        self._date = date.today().strftime("%Y.%m.%d")

        # Debounce bookkeeping.
        self._stable_key = None
        self._stable_count = 0
        self._evaluated_key = None
        self._verified_start = False
        self._frames = 0

        self.move_count = 0
        self.last_san = None

        self._write_files()  # emit starting position immediately

    # -- public API -------------------------------------------------------

    def update(self, board_dict: dict) -> None:
        """Feed one frame's detection, `{square_name: (label, conf)}`.

        Does nothing until the same board state has been seen for
        `stability_frames` frames, at which point it tries to record a move."""
        observed = self._to_observed(board_dict)
        newly_stable = self._advance_debounce(frozenset(observed.items()))

        if self.debug and self._frames % 30 == 0:
            self._log(
                f"[rec/dbg] frame {self._frames}: {len(observed)} pieces detected, "
                f"current state stable for {self._stable_count}/"
                f"{self.stability_frames} frames"
            )

        if newly_stable:
            self._on_stable(observed)

    def update_changed(self, changed_names: set[str]) -> bool:
        """Feed one frame's *changed-square* set (model-free "diff" mode).

        `changed_names` is the set of square names whose pixels differ from the
        board as of the last committed move. Once that set has been stable for
        `stability_frames` frames, the move whose own squares it matches is
        recorded. Returns True iff a move was committed this frame, so the caller
        can refresh its reference image to the new position."""
        changed = self._names_to_squares(changed_names)
        newly_stable = self._advance_debounce(frozenset(changed))

        if self.debug and self._frames % 30 == 0:
            self._log(
                f"[rec/dbg] frame {self._frames}: {len(changed)} squares changed "
                f"{self._fmt_squares(changed)}, stable for {self._stable_count}/"
                f"{self.stability_frames} frames"
            )

        if newly_stable:
            move = self._infer_move_from_changed(changed)
            if move is not None:
                self._commit(move)
                return True
        return False

    def _advance_debounce(self, key) -> bool:
        """Update the stability counter for `key`; return True exactly once, on
        the frame a state first becomes stable (shared by both update paths)."""
        self._frames += 1
        if key == self._stable_key:
            self._stable_count += 1
        else:
            self._stable_key = key
            self._stable_count = 1

        if self._stable_count == self.stability_frames and key != self._evaluated_key:
            self._evaluated_key = key
            return True
        return False

    def finalize(self) -> None:
        """Flush the final game state (e.g. when the user stops recording)."""
        self._write_files()
        result = self.board.result(claim_draw=True)
        self._log(f"[rec] finalized: {self.move_count} moves, result {result}")

    # -- internals --------------------------------------------------------

    @staticmethod
    def _to_observed(board_dict: dict) -> dict:
        """Detection dict -> `{square_index: piece_symbol}` for occupied squares."""
        observed = {}
        for sq_name, (label, _conf) in board_dict.items():
            symbol = label_to_symbol(label)
            if symbol is None:
                continue
            try:
                square = chess.parse_square(sq_name)
            except ValueError:
                continue
            observed[square] = symbol
        return observed

    @staticmethod
    def _match_score(board: chess.Board, observed: dict) -> int:
        """Count how many of the 64 squares agree between a candidate position
        and the detection. Empty-on-both counts as agreement; undetected squares
        are treated as empty."""
        score = 0
        for square in range(64):
            piece = board.piece_at(square)
            candidate = piece.symbol() if piece else None
            if candidate == observed.get(square):
                score += 1
        return score

    def _on_stable(self, observed: dict) -> None:
        if not self._verified_start:
            self._verified_start = True
            start_score = self._match_score(self.board, observed)
            if start_score < self.min_match:
                self._log(
                    f"[rec] warning: detected board only matches {start_score}/64 "
                    "squares of the recording's start position. Check piece "
                    "placement and the board orientation (the 'f' flip toggle)."
                )

        move = self._infer_move(observed)
        if move is not None:
            self._commit(move)

    def _infer_move(self, observed: dict) -> chess.Move | None:
        """Pick the legal move whose result best matches `observed`, or None if
        no move clearly explains the change."""
        baseline = self._match_score(self.board, observed)

        scored = []
        for move in self.board.legal_moves:
            self.board.push(move)
            scored.append((self._match_score(self.board, observed), move))
            self.board.pop()

        if not scored:
            return None  # game over, no legal moves

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_move = scored[0]

        if self.debug:
            self._log(
                f"[rec/dbg] stable board ({len(observed)} pieces): current "
                f"position matches {baseline}/64, best legal move "
                f"{self.board.san(best_move)} would match {best_score}/64 "
                f"(needs > {baseline} and >= {self.min_match})"
            )

        # The board must actually have changed in a way a move explains better
        # than doing nothing, and the match must be plausibly a real position.
        if best_score <= baseline or best_score < self.min_match:
            if self.debug:
                reason = (
                    "no legal move beats keeping the current position"
                    if best_score <= baseline
                    else f"best match {best_score}/64 is below min_match "
                    f"{self.min_match}/64"
                )
                self._log(f"[rec/dbg] no move recorded: {reason}")
            return None

        # Reject ties: two legal moves explaining the detection equally well
        # means the view is ambiguous (e.g. a missed piece). Wait for a clearer
        # frame rather than guess.
        if len(scored) > 1 and scored[1][0] == best_score:
            self._log(
                f"[rec] ambiguous detection (tie at {best_score}/64); "
                "waiting for a clearer view"
            )
            return None

        return best_move

    @staticmethod
    def _names_to_squares(names: set[str]) -> set[int]:
        """Square names -> `{square_index}`, dropping anything unparseable."""
        squares = set()
        for name in names:
            try:
                squares.add(chess.parse_square(name))
            except ValueError:
                continue
        return squares

    @staticmethod
    def _fmt_squares(squares: set[int]) -> str:
        return "{" + ", ".join(sorted(chess.square_name(s) for s in squares)) + "}"

    def _altered_squares(self, move: chess.Move) -> set[int]:
        """Squares whose occupant changes when `move` is played — i.e. the
        squares an image diff should light up. Covers the from/to squares plus
        the rook in castling and the captured pawn in en passant."""
        before = self.board.piece_map()
        self.board.push(move)
        after = self.board.piece_map()
        self.board.pop()
        return {
            sq for sq in set(before) | set(after) if before.get(sq) != after.get(sq)
        }

    def _infer_move_from_changed(self, changed: set[int]) -> chess.Move | None:
        """Pick the legal move whose own squares the observed change set covers.

        A move only qualifies once *every* square it alters shows change (so a
        piece merely lifted off its origin, with the destination not yet filled,
        matches nothing), while up to `diff_tolerance` extra changed squares are
        tolerated as noise. Promotion piece type can't be read from occupancy,
        so an otherwise-unique promotion defaults to a queen; any other tie is
        treated as ambiguous and skipped."""
        if not changed:
            return None

        best_extra = None
        best_moves: list[chess.Move] = []
        for move in self.board.legal_moves:
            altered = self._altered_squares(move)
            if not altered.issubset(changed):
                continue
            extra = len(changed - altered)
            if extra > self.diff_tolerance:
                continue
            if best_extra is None or extra < best_extra:
                best_extra, best_moves = extra, [move]
            elif extra == best_extra:
                best_moves.append(move)

        if not best_moves:
            self._log(
                f"[rec] change {self._fmt_squares(changed)} matches no legal move — "
                "press 'v' to validate against the model"
            )
            return None

        if len(best_moves) > 1:
            from_to = {(m.from_square, m.to_square) for m in best_moves}
            if len(from_to) == 1 and all(m.promotion for m in best_moves):
                if self.debug:
                    self._log("[rec/dbg] promotion ambiguous; defaulting to queen")
                return next(m for m in best_moves if m.promotion == chess.QUEEN)
            self._log(
                f"[rec] ambiguous change {self._fmt_squares(changed)} "
                f"({len(best_moves)} legal moves fit); waiting for a clearer view"
            )
            return None

        return best_moves[0]

    def _commit(self, move: chess.Move) -> None:
        # SAN and the move label must both be read before the move is pushed.
        label = self._move_label(self.board)
        san = self.board.san(move)
        self.board.push(move)
        self.move_count += 1
        self.last_san = san
        self._log(f"[rec] {label} {san}   {self.board.fen()}")
        self._write_files()
        if self.board.is_game_over():
            outcome = self.board.outcome(claim_draw=True)
            reason = outcome.termination.name.lower() if outcome else "unknown"
            self._log(f"[rec] game over: {self.board.result()} ({reason})")

    @staticmethod
    def _move_label(board: chess.Board) -> str:
        """Move number label for the side to move, e.g. '1.' (White) or '12...'
        (Black). Read from the board *before* the move is pushed, so it stays
        correct even when recording starts mid-game with Black to move."""
        dots = "." if board.turn == chess.WHITE else "..."
        return f"{board.fullmove_number}{dots}"

    def _build_game(self) -> chess.pgn.Game:
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = "Live capture"
        game.headers["Site"] = "chessvision"
        game.headers["Date"] = self._date
        game.headers["White"] = "?"
        game.headers["Black"] = "?"
        game.headers["Result"] = self.board.result(claim_draw=True)
        return game

    def _write_files(self) -> None:
        self.pgn_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pgn_path, "w") as f:
            print(self._build_game(), file=f, end="\n\n")

        lines = [
            f"# Live capture {self._date}",
            f"# {'move':<6} {'SAN':<8} FEN",
            f"  {'start':<6} {'':<8} {self._start_fen}",
        ]
        replay = chess.Board(self._start_fen)
        for move in self.board.move_stack:
            label = self._move_label(replay)
            san = replay.san(move)
            replay.push(move)
            lines.append(f"  {label:<6} {san:<8} {replay.fen()}")
        self.fen_path.write_text("\n".join(lines) + "\n")
