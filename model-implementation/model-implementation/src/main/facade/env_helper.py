class EnvironmentVariable:
    
    __key_value_pair = {};


    
    # read_env
    # @private
    # This function will read .env file, and extracting all values as dictionary of self.key_value_pair
    def _read_env(self):
        
        try:
            fopen = open("../../../.env", "r");
        
            for line in fopen:
                key, value = line.split("=");
                self.__key_value_pair[key] = value;       
        except:
            print("Failed to retrieve environment variable. Ensure that your .env is exist as /model-implementation/model-implementation/.env");

    # get_key
    # @public
    # This function will get a key value from key_value_pair, and return it as string.
    # If there are no value exist for the searched key, this function will throw an exception.
    def get_key(self, key):
        try:
            self.__key_value_pair[key];
        except:
            raise Exception("Make sure the provided key is already registered in you .env!");