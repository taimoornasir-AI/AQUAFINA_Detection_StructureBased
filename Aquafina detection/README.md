# Aquafina camera detection

This prototype uses OpenCV ORB features, homography, and a normalized shape signature to decide whether a bottle in a live camera frame resembles the supplied reference. The Tkinter UI provides clear feedback and explicit labeling. Camera work runs in a background thread so the UI stays responsive.

## Run

From this folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Run `python app.py --camera 1` if the desired camera is not index 0. Close the UI window normally.

When the result is wrong, use **Confirm Aquafina** or **Mark not Aquafina**. Images are saved under `feedback/` with labels. Confirmed Aquafina images become additional reference embeddings on the next run, helping across bottle sizes, angles, and lighting.

## Persistent embeddings

`aquafina_embeddings.sqlite3` stores ORB descriptor vectors, keypoint coordinates, normalized shape vectors, labels, and image dimensions as SQLite BLOBs. The reference image is inserted once; later executions load the vectors instead of rebuilding a model. Labeled feedback is inserted immediately and reused in the next frame and next execution. This is an embedding index, not retraining on every camera frame.

## Important limitation

The app does not silently train on every frame. That would store thousands of near-identical or incorrectly labeled images and degrade accuracy. It collects only explicit user-labeled examples. The normalized shape signature supports scale changes such as 500 ml and 1 L, but reliable production coverage still needs many labeled bottle sizes and viewpoints.

The supplied `.mtl` file is read for its `Kd` material colors. It does not contain vertices, faces, or geometry, so it cannot by itself create camera embeddings. A paired `.obj` file would be needed for 3D geometry; the current reference image and user-labeled camera frames remain the actual visual sources.

## Production recommendation

Collect several hundred positive images of Aquafina bottles and negative images of other bottles, with different backgrounds, rotations, lighting, and distances. Train a small YOLO detector using Ultralytics, export it to ONNX, and run inference with OpenCV DNN or ONNX Runtime. Keep this OpenCV camera loop as the capture/display layer and replace `find_bottle` with the detector inference once the dataset exists.