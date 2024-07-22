import time

# Function to write a message to a file
def write_to_file(filename, message):
    with open(filename, 'w') as file:
        file.write(message)

# Main script
if __name__ == "__main__":
    # Sleep for 20 seconds
    time.sleep(5)
    
    # Define the filename and message
    filename = 'output.txt'
    message = 'This message is written to the file after a 20 second delay.'
    
    # Write the message to the file
    write_to_file(filename, message)
    
    # Print confirmation
    print(f"Message written to {filename}")
