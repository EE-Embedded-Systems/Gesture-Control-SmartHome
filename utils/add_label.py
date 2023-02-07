import csv

GESTURE = "one"

# Open the CSV file for reading
with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/data/"+GESTURE+".csv", "r") as input_file:
    # Create a CSV reader object
    reader = csv.reader(input_file)

    # Open the output file for writing
    with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/data/"+GESTURE+"2.csv", "w", newline='') as output_file:
        # Create a CSV writer object
        writer = csv.writer(output_file)

        # Iterate over the rows in the input file
        for row in reader:
            # Add the desired word to the end of the row
            row.append("one")

            # Write the modified row to the output file
            writer.writerow(row)
