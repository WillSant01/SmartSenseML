import json
import requests
import os


def user_login(url, user_data_path):
    """
    This function allow user to login if user info already exists otherwise ask for registration
    :param url: server url
    :param user_data_path: path to the user info file
    :return: login status
    """
    if not os.path.exists(user_data_path):
        # update_registration(user_data_path)
        pass
    else:
        with open(user_data_path) as json_file:
            user_data = json.load(json_file)
        # sending login request to server
        res = requests.post(url + "login", data=json.dumps(user_data))
        res = json.loads(res.content)

        if res["status"] == "wrongpassword":
            print("Your password is worng, please enter again")
            update_local_password(user_data_path, user_data)
            return res
        elif res["status"] == "wronguser":
            print("your user name does not exists, please register")
            update_local_registration(user_data_path)
            return res
        elif res["status"] == "successful":
            print("Login successful")
            return res
        else:
            print("register again")
            update_local_registration(user_data_path)
            return res


def update_local_password(user_data_path, user_data):
    """
    This function allow user to change password if it is wrong
    :param user_data_path: path of the user info file
    :param user_data: current user data
    :return: None
    """
    print("Please enter correct password:")
    password = input()
    user_data["password"] = password
    with open(user_data_path, "w") as outfile:
        json.dump(user_data, outfile)


def update_local_registration(user_data_path):
    """
    this function update local file with user's login information
    :param user_data_path: path of the file where user info will be stored
    :return: None
    """
    print("Enter your first name")
    first_name = input()
    print("Enter your second name")
    second_name = input()
    print("Enter a password")
    password = input()
    print("Enter your preferred user name(all lower case)")
    user_name = input()
    user_data = {
        "first_name": str(first_name),
        "last_name": str(second_name),
        "password": str(password),
        "user_id": str(user_name).lower(),
    }
    with open(user_data_path, "w") as outfile:
        json.dump(user_data, outfile)
