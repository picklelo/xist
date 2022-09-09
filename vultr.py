import requests
import time
import os


API_KEY = os.getenv("VULTR_API_KEY")
base = "https://api.vultr.com/v2"
headers = {"Authorization": f"Bearer {API_KEY}"}

DB_NAME = "memories"
DB_APP_NAME = "Docker on Ubuntu 20.04 x64"
DB_MONTHLY_COST = 5

DREAMER_NAME = "dream"
DREAMER_APP_NAME = "NVIDIA Docker on Ubuntu 22.04 LTS"
DREAMER_MONTHLY_COST = 180

REGION = "ewr"
SSH_KEY_NAME = "MacBook"


def get(endpoint, **kwargs):
    args = "&".join([f"{k}={v}" for k, v in kwargs.items()])
    url = f"{base}/{endpoint}?{args}"
    return requests.get(url, headers=headers).json()


def post(endpoint, **kwargs):
    body = {k: v for k, v in kwargs.items()}
    url = f"{base}/{endpoint}"
    return requests.post(url, headers=headers, json=body)


def delete(endpoint, **kwargs):
    body = {k: v for k, v in kwargs.items()}
    url = f"{base}/{endpoint}"
    print("Deleting", url)
    return requests.delete(url, headers=headers, json=body)


def filt(collection, key, value):
    items = [item for item in collection if item[key] == value]
    if len(items) > 0:
        return items[0]
    return None

def getr(resource, key = None, value = None, **kwargs):
    vals = get(resource, **kwargs)[resource.replace("-", "_")]
    if key is None and value is None:
        return vals
    return filt(vals, key, value)

def create_instance(type: str, monthly_cost: int, app_name: str, label: str, script: str):
    # Get the plan.
    plan = getr("plans", "monthly_cost", monthly_cost, type=type)
    assert REGION in plan["locations"]
    plan_id = plan["id"]

    # Get the app.
    app_id = getr("applications", "deploy_name", app_name)["id"]

    # Get the script.
    script_id = getr("startup-scripts", "name", script)["id"]

    # Get the ssh key.
    sshkey_id = getr("ssh-keys", "name", SSH_KEY_NAME)["id"]

    # Create the instance.
    print(f"Creating {label} with plan {plan_id} in region {REGION} with app {app_id} and script {script_id}")
    response = post(
        "instances",
        plan=plan_id,
        region=REGION,
        app_id=app_id,
        hostname=label,
        label=label,
        script_id=script_id,
        sshkey_id=[sshkey_id],
    ).json()

    assert "status" not in response, "Failed to create instance"
    return response

def stop_instance(name: str):
    instance = get_instance(name)
    assert instance is not None, "Instance not found!"
    print("Stopping instance", instance["id"])
    return delete(f"instances/{instance['id']}")
    

def create_dream():
    instance_id = len(getr("instances")) + 1
    return create_instance(
        type="vcg",
        monthly_cost=DREAMER_MONTHLY_COST,
        app_name=DREAMER_NAME,
        label=f"{DREAMER_NAME}{instance_id}",
        script=DREAMER_NAME,
    )

def is_active(resource):
    return resource["status"] == "active"

def wait_for_resource(fn):
    while True:
        resource = fn()
        if resource is not None and is_active(resource):
            break
        print("Waiting for resource to be ready.")
        time.sleep(5)
    print("Created resource", resource)
    return resource

def get_block(name: str):
    return getr("blocks", "label", name)

def get_instance(name: str):
    return getr("instances", "label", name)

def create_block(name: str, size_gb: int=10):
    # Create the block if necessary.
    if get_block(name) is None:
        print(f"Creating block {name}")
        post("blocks", label=name, size_gb=size_gb, region=REGION)

    # Wait for the block to be ready.
    return wait_for_resource(lambda: get_block(name))

def attach_block(block_name: str, instance_name: str):
    block = get_block(block_name)
    assert block is not None, "Block not found"
    assert block["attached_to_instance"] == "", "Block already attached!"

    # Get the instance, wait for it to be ready.
    instance = wait_for_resource(lambda: getr("instances", "label", instance_name))
    response = post(f"blocks/{block['id']}/attach", instance_id=instance["id"])
    return response

def create_db():
    if get_instance(DB_NAME) is None:
        create_instance(
            type="vc2",
            monthly_cost=DB_MONTHLY_COST,
            app_name=DB_APP_NAME,
            label=DB_NAME,
            script=DB_NAME,
        )
        print("Waiting for db to be ready.")
        time.sleep(10)
    if get_block(DB_NAME) is None:
        create_block(DB_NAME)
    attach_block(DB_NAME, DB_NAME)

