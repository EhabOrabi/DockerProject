#!/bin/bash

echo "Initializing MongoDB replica set..."

# Start MongoDB server with replica set configuration
mongod --replSet myReplicaSet --bind_ip localhost,mongo_primary &

# Wait for MongoDB to start
sleep 10

echo "Initializing MongoDB replica set..."
mongo --host mongo_primary:27017 <<EOF
rs.initiate({
  _id: "myReplicaSet",
  version: 1,
  members: [
    { _id: 0, host: "mongo_primary:27017" },
    { _id: 1, host: "mongo_1:27017" },
    { _id: 2, host: "mongo_2:27017" }
  ]
})
EOF

# Check if replica set initialization was successful
if [ $? -eq 0 ]; then
  echo "MongoDB replica set initialized successfully."
else
  echo "Error: MongoDB replica set initialization failed."
fi
