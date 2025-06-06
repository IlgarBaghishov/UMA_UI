"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import hashlib
import os
from pathlib import Path

import ase
import backoff
import gradio as gr
import huggingface_hub as hf_hub
import requests
from ase.calculators.calculator import Calculator
from ase.db.core import now
from ase.db.row import AtomsRow
from ase.io.jsonio import decode, encode
from requests.exceptions import HTTPError


def hash_save_file(atoms: ase.Atoms, task_name, path: Path | str):
    atoms = atoms.copy()
    atoms.info["task_name"] = task_name
    atoms.write(
        Path(path)
        / f"{hashlib.md5(atoms_to_json(atoms).encode('utf-8')).hexdigest()}.traj"
    )
    return


def validate_uma_access(oauth_token):
    try:
        hf_hub.HfApi().auth_check(repo_id="facebook/UMA", token=oauth_token.token)
        return True
    except (hf_hub.errors.HfHubHTTPError, AttributeError):
        return False


class HFEndpointCalculator(Calculator):
    # A simple calculator that uses the Hugging Face Inference Endpoints to run

    implemented_properties = ["energy", "free_energy", "stress", "forces"]

    def __init__(
        self,
        atoms,
        endpoint_url,
        oauth_token,
        task_name,
        example=False,
        *args,
        **kwargs,
    ):
        # If we have an example structure, we don't need to check for authentication
        # Otherwise, we need to check if the user is authenticated and has gated access to the UMA models
        if not example:
            if validate_uma_access(oauth_token):
                try:
                    hash_save_file(atoms, task_name, "/data/custom_inputs/")
                except FileNotFoundError:
                    pass
            else:
                raise gr.Error(
                    "You need to log in to HF and have gated model access to UMA before running your own simulations!"
                )

        self.url = endpoint_url
        self.token = os.environ["HF_TOKEN"]
        self.atoms = atoms
        self.task_name = task_name

        super().__init__(*args, **kwargs)

    @staticmethod
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException,),
        max_tries=10,
        jitter=backoff.full_jitter,
    )
    def _post_with_backoff(url, headers, payload):
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        task_name = self.task_name.lower()

        payload = {
            "inputs": atoms_to_json(atoms, data=atoms.info),
            "properties": properties,
            "system_changes": system_changes,
            "task_name": task_name,
        }

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        print(payload)

        try:
            response = self._post_with_backoff(self.url, headers, payload)
            response_dict = response.json()
        except HTTPError as error:
            hash_save_file(atoms, task_name, "/data/custom_inputs/errors/")
            raise gr.Error(
                f"Backend failure during your calculation; if you have continued issues please file an issue in the main FAIR chemistry repo (https://github.com/facebookresearch/fairchem).\n{error}"
            )

        # Load the response and store the results in the calc and atoms object
        response_dict = decode(response_dict)
        self.results = response_dict["results"]
        atoms.info = response_dict["info"]


def atoms_to_json(atoms, data=None):
    # Similar to ase.db.jsondb

    mtime = now()

    row = AtomsRow(atoms)
    row.ctime = mtime

    dct = {}
    for key in row.__dict__:
        if key[0] == "_" or key in row._keys or key == "id":
            continue
        dct[key] = row[key]

    dct["mtime"] = mtime

    if data:
        dct["data"] = data
    else:
        dct["data"] = {}

    constraints = row.get("constraints")
    if constraints:
        dct["constraints"] = constraints

    return encode(dct)
