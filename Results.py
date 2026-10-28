import csv

def save_csv(filename, measurements):

    if not measurements:
        return

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=measurements[0].keys()
        )

        writer.writeheader()
        writer.writerows(measurements)