# Ninja Cam ✌️💨

Make a peace sign on camera and vanish in a puff of smoke, leaving an empty chair.
Make it again and you pop back in. Works in Teams, Zoom, Meet and anything else that
lets you pick a webcam.

## Setup (Windows 11, once)

1. Install **Python 3.12** from https://www.python.org/downloads/windows/
   (3.10 or 3.11 also work; 3.13+ does not, because mediapipe has no build for it).
2. Install **OBS Studio** from https://obsproject.com. You don't need to open it. It only
   provides the "OBS Virtual Camera" device that Teams will see.
3. Double-click `setup.bat`.

## Use it

1. Double-click `run.bat`. A preview window opens.
2. **Press `B` and step out of frame for ~5 seconds.** This records your empty background
   so the empty chair looks perfect. If you skip it, the app guesses what's behind you
   and the result is blurrier.
3. In Teams: Settings → Devices → Camera → **OBS Virtual Camera**.
4. Time to say bye: hold up a ✌️ for half a second. 💨
5. Do it again to come back.

| Key   | Action                              |
|-------|-------------------------------------|
| B     | capture a clean background          |
| SPACE | vanish / come back without gesture  |
| M     | mirror the preview                  |
| Q     | quit                                |

The preview is mirrored like a selfie camera. Teams gets the normal (unmirrored)
image without the text overlay.

## Options

```
run.bat --camera 1          # use another webcam
run.bat --width 1920 --height 1080
run.bat --no-preview
```

## Troubleshooting

- **"Could not open camera"**: close any app that is using the webcam (including the
  Teams camera preview), or try `--camera 1`.
- **"No virtual camera"**: OBS isn't installed. Install it, then restart `run.bat`.
- **Teams shows the OBS logo/placeholder**: `run.bat` isn't running.
- **Triggers too easily / not easily enough**: adjust `PEACE_HOLD_SEC` near the top of
  `ninja_cam.py`.
