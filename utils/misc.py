import gc
from names_generator import generate_name
import hashlib

# dummy gc callback for optuna, its probably useless for memory management
def dummy_gc(input1, input2):
    gc.collect()

# random name generator

def hash_string_to_8_digits(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    return int(hex_digest, 16) % (10**8)

def random_name(config, X, y):
    return(generate_name(seed=hash_string_to_8_digits(str(config))+hash_string_to_8_digits(str(X))+hash_string_to_8_digits(str(y))))