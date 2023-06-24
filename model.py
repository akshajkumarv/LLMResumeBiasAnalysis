class ModelLoader:
    def __init__(self, args):
        self.model_name = args.model_name

        if self.model_name == 'gpt':
            print("Loading GPT Model")
            # read api key from given file path
            with open(f"api_keys/{self.model_name}/{args.api_key_file}", 'r') as file:
                # first line is org key, second line is api key
                self.organization = file.readline().replace('\n', '')
                self.api_key = file.readline().replace('\n', '')

        elif self.model_name == 'bard':
            print("Loading Bard Model")
            # read api key from given file path
            with open(f"api_keys/{self.model_name}/{args.api_key_file}", 'r') as file:
                self.api_key = file.read().replace('\n', '')
            
        elif self.model_name == 'claude':
            print("Loading Claude Model")
            with open(f"api_keys/{self.model_name}/{args.api_key_file}", 'r') as file:
                self.client = file.read().replace('\n', '')
            
        else:
            raise ValueError(f"Model {self.model_name} not supported")

