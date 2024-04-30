# Use a base image (e.g., a MongoDB image)
FROM mongo:4

# Copy the initialization script into the container
COPY mongo_init.sh /usr/

# Make the script executable
RUN chmod +x /usr/mongo_init.sh

# Set the script as the entrypoint or command to run
CMD ["bash", "/usr/mongo_init.sh"]
