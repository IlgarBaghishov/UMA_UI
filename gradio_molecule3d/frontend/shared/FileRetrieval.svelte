<script lang="ts">
  import { createEventDispatcher, tick, getContext } from "svelte";

  import { FileData } from "@gradio/client";
  import LoadingSpinner from "./loading_spinner.svelte";
  let uploaded_files;
  const dispatch = createEventDispatcher();

  import type { Client } from "@Gradio Uptime/client";
  export let upload: Client["upload"];

  export let root: string;
  async function handle_upload(file_data: FileData): Promise<void> {
    await tick();
    const upload_id = Math.random().toString(36).substring(2, 15);
    uploaded_files = await upload([file_data], root, upload_id);
    dispatch("load", uploaded_files[0]);
  }
  let loading = false;
  async function fetchFromDB(identifier, database): Promise<void> {
    let dbs = {
      pdb_assym: {
        url: "https://files.rcsb.org/view/",
        ext: ".pdb",
      },
      pdb_bioass: {
        url: "https://files.rcsb.org/view/",
        ext: ".pdb1",
      },
      af: {
        url: "https://alphafold.ebi.ac.uk/files/AF-",
        ext: "-F1-model_v4.pdb",
      },
      esm: {
        url: "https://api.esmatlas.com/fetchPredictedStructure/",
        ext: ".pdb",
      },
      // pubchem: "pubchem",
      // text: "text",
    };
    let url = dbs[database]["url"];
    let ext = dbs[database]["ext"];
    // load the file and save blob
    // emulate file upload by fetching from the db and triggering upload
    // check if response status is 200, then return blob
    loading = true;
    let file = null;
    try {
      file = await fetch(url + identifier + ext)
        .then((r) => {
          loading = false;
          if (r.status == 200) {
            return r.blob();
          } else {
            dispatch("notfound");
          }
        })
        .then((blob) => {
          return new File([blob], identifier + ".pdb", { type: "text/plain" });
        });
    } catch (error) {
      loading = false;
      dispatch("notfound");
    }
    let file_data = new FileData({
      path: identifier + ".pdb",
      orig_name: identifier + ".pdb",
      blob: file,
      size: file.size,
      mime_type: file.type,
      is_stream: false,
    });
    await handle_upload(file_data);
  }
  let selectedValue = "pdb_assym";
  let placeholder = "";
  let textInput = "";
  function handleSelect(event) {
    selectedValue = event.target.value;
  }
  let placeholders = {
    pdb_assym: "Enter PDB identifier",
    pdb_bioass: "Enter PDB identifier",
    af: "Enter UniProt identifier",
    esm: "Enter MGnify protein identifier",
    // pubchem: "Enter PubChem identifier",
    // text: "Enter molecule in PDB or mol2 format",
  };
  $: placeholder = placeholders[selectedValue];
  function isEnterPressed(event) {
    if (event.key === "Enter") {
      fetchFromDB(textInput, selectedValue);
    }
  }
</script>

<style>
  .py {
    padding: 10px 0;
  }
  .btn {
    margin: 0 5px;
    padding: 3px 15px;
    border: var(--button-border-width) solid
      var(--button-secondary-border-color);
    background: var(--button-secondary-background-fill);
    color: var(--button-secondary-text-color);
    border-radius: var(--button-large-radius);
    padding: var(--button-large-padding);
    font-weight: var(--button-large-text-weight);
    font-size: var(--button-large-text-size);
    cursor: pointer;
  }
  .btn:hover {
    border-color: var(--button-secondary-border-color-hover);
    background: var(--button-secondary-background-fill-hover);
    color: var(--button-secondary-text-color-hover);
    box-shadow: var(--button-shadow-hover);
  }
  .or {
    color: var(--body-text-color-subdued);
    text-align: center;
    display: block;
  }
  .wfull {
    width: 100%;
  }
  .mt-2 {
    margin-top: 2rem;
  }
  .input {
    box-shadow: var(--input-shadow);
    background: var(--input-background-fill);
    border: var(--input-border-width) solid var(--input-border-color);
    border-radius: var(--input-radius);
    margin: 0 5px;
  }
  .select {
    outline: none;
    border: none;
  }
  .flex {
    display: flex;
    justify-content: space-between;
  }
  .inp {
    width: 100%;
    border: 0 #fff !important;
    outline: none !important;
  }
  .inp:focus,
  .inp:hover {
    border: 0 !important;
    outline: none !important;
  }
  .text-center {
    text-align: center;
  }
</style>
