# Setting System Volume to 50%

It is important to run the AFHI web demos using a consistent volume. The system has been calibrated, and your results will be most accurate if you use identical settings to those used during calibration. The following setup was used for calibration:

* MacBook Pro with volume set to 50%
* Google Pixel Buds Pro 2 in-ear headphones

We have found good results can also be obtained using the Windows operating system and with other Bluetooth headphones (such as the Apple AirPods Pro 2), but results may vary. However, what is critical in all cases is that you set your volume to 50%.

## Volume control on macOS

We recommend two methods to **set** the system volume to exactly 50%:

1. Use the volume control buttons on the keyboard until the volume popup shows that the volume is set to the middle setting (this is likely 8 button presses away from 0 or 100%).
2. Alternatively, run the following in terminal:
   ```
   osascript -e 'set volume output volume 50'
   ```

There are also two good approaches to **check** that the system volume really is at 50%:

1. Use the **Audio MIDI Setup** app that is built-in to macOS.
   1. Click on the headphones that you're using.
   2. Check that the "Value" listed beside each of the two sliders is 0.5.
2. Alternatively, run the following in terminal and confirm that the result is `50`:
   ```
   osascript -e 'output volume of (get volume settings)'
   ```
