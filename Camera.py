from picamera2 import Picamera2
from libcamera import controls
from pathlib import Path

def take_picture(camera, counter, num_imgs, batch, lux):

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

    try:
        for i in range(num_imgs):
            camera.capture_file(f"Datasets/{batch}/{batch}{counter+1:03d}_{lux}.jpg")
            counter += 1

    finally:
        camera.stop()

    return counter

if __name__ == "__main__":

    camera = Picamera2()
    counter = 0  

    try:
        while True:
            try:
                input_num_imgs = int(input("Enter the number of images to capture: "))

                if input_num_imgs == 0:
                    break

                if input_num_imgs < 0:
                    print("Please enter a positive number.")
                    continue

                input_batch = input("Enter the batch name: ")

                if not input_batch:
                    print("Batch name cannot be empty.")
                    continue

                if not Path(f"Datasets/{input_batch}").exists():
                    print(f"Batch '{input_batch}' does not exist. Please enter a valid batch name.")
                    continue

                lux = (input("Enter the lux value ('low', 'med', 'high'): "))

                if lux not in ['low', 'med', 'high']:
                    print("Invalid lux value. Please enter 'low', 'med', or 'high'.")
                    continue

                counter = take_picture(camera, counter, input_num_imgs, input_batch, lux)

            except ValueError:
                print("Invalid input. Please enter a valid number or string.")
            except KeyboardInterrupt:
                print("\nExiting the program.")
                break

    finally:
        camera.close()
