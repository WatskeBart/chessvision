"""chessvision — track a live chess game from a single overhead camera.

Layout:
  chessvision.core      shared library: settings-driven board detection,
                        occupancy diffing, and game tracking.
  chessvision.app       live applications (piece detection + recording, camera
                        preview).
  chessvision.training  dataset capture, auto-labeling, training and export.
"""
