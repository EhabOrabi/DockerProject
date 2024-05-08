#!/bin/bash

# Function to initialize the replica set
#

sleep 10
initialize_replica_set() {
  mongo --host "$1" <<EOF
  rs.initiate({
    _id: "myReplicaSet",
    version: 1,
    members: [
      { _id: 0, host: "mongo_1:27017", priority: 2 },
      { _id: 1, host: "mongo_2:27017", priority: 1 },
      { _id: 2, host: "mongo_3:27017", priority: 1 }
    ]
  })
EOF
}

# Initialize the replica set on mongo_1:27017
initialize_replica_set "mongo_1:27017"

