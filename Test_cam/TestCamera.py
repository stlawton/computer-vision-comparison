from time import sleep
from picamera2 import Picamera2
from libcamera import controls

camera = Picamera2()

config = camera.create_still_configuration(
    main={
        "size": (1920, 1080),
        "format": "BGR888"
    },
    controls={
    "AeEnable": False,
    "ExposureTime": 30000,
    "AnalogueGain": 1.787,

    "AwbEnable": False,
    "ColourGains": (1.6718, 2.0683),

    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 1.0,

    "FrameDurationLimits": (66667, 66667)
    }
)
camera.configure(config)

camera.start()
sleep(2)
camera.capture_file("config_test1.jpg")
sleep(2)
camera.capture_file("config_test2.jpg")
sleep(2)
camera.capture_file("config_test3.jpg")

for i in range(5):
    metadata = camera.capture_metadata()

    print("ExposureTime:", metadata.get("ExposureTime"))
    print("AnalogueGain:", metadata.get("AnalogueGain"))
    print("DigitalGain:", metadata.get("DigitalGain"))
    print("ColourGains:", metadata.get("ColourGains"))
    print("LensPosition:", metadata.get("LensPosition"))
    print("FrameDuration:", metadata.get("FrameDuration"))

    sleep(10)

camera.stop()