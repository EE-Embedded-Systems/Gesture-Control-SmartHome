import csv

# Open the CSV file for reading
with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/Data/thumbs_up.csv", "r") as input_file:
    # Create a CSV reader object
    reader = csv.reader(input_file)
    
    # Open the output file for writing
    with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/Data/thumbs_up2.csv", "w", newline='') as output_file:
        # Create a CSV writer object
        writer = csv.writer(output_file)
        
        # Iterate over the rows in the input file
        for row in reader:
            # Add the desired word to the end of the row
            row.append("fist")
            
            # Write the modified row to the output file
            writer.writerow(row)