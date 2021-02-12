import yaml


class CustomNamespace:
    def __init__(self, **kwargs):

        complement = {}
        for k, v in kwargs.items():
            if k.islower():
                complement[k.upper()] = v
            else:
                complement[k.lower()] = v

        self.__dict__.update(kwargs)
        self.__dict__.update(complement)

    def __repr__(self):
        return 'Options: ' + ', '.join('{}={}'.format(k, v) for k, v in self.__dict__.items() if k.isupper())


with open('config.yaml') as f:
    options = yaml.safe_load(f)

training = CustomNamespace(**options['training'])
data = CustomNamespace(**options['data'])
