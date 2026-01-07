import json
import base64


def load_token_from_file(filename):
    with open(filename, "rt") as fin:
        return fin.read().strip()


def decode_jwt_token_base64(token_str):
    spl_token_str = token_str.split(".")
    token_sub_str = spl_token_str[1] + b"=" * (-len(spl_token_str[1]) % 4)
    decoded_user_info = json.loads(token_sub_str.decode("base64"))
    return decoded_user_info


if __name__ == "__main__":
    token_str = load_token_from_file("./token.txt")
    dec_jwt_t = decode_jwt_token_base64(token_str)
