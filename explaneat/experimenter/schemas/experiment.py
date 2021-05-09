data_location = {
    "type": "object",
    "properties": {
        "xs": {
            "type": "string"
        },
        "ys": {
            "type": "string"
        }
    }
}

experiment = {
    "type":"object",
    "properties":{
        "experiment": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "description": {
                    "type": "string"
                },
                "codename": {
                    "type": "string"
                }
            },
            "required":[
                "name",
                "description",
                "codename"
            ]
        },
        "results": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string"
                }
            },
            "required":["location"]
        },
        "data": {
            "type": "object",
            "properties": {
                "locations": {
                    "train":data_location,
                    "test": data_location
                }
            }
        },
        "model": {
            "type":"object",
            "properties":{
                "config_file": {
                    "type": "string"
                }
            }
        }
    }
}