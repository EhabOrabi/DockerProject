# Use a base image (e.g., a MongoDB image)
FROM mongo:4

# Copy initialization and user creation scripts into the container
COPY mongo_init.sh /usr/

# Make the scripts executable
RUN chmod +x /usr/mongo_init.sh

# Set the scripts as the entrypoint or command to run
CMD ["bash","/usr/mongo_init.sh"]

