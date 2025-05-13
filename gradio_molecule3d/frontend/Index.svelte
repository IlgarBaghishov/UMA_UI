<svelte:options accessors={true} />

<script context="module" lang="ts">
  export { default as FilePreview } from "./shared/FilePreview.svelte";
  export { default as BaseFileUpload } from "./shared/FileUpload.svelte";
  export { default as BaseFile } from "./shared/File.svelte";
  export { default as BaseExample } from "./Example.svelte";
</script>

<script lang="ts">
  import "./style.css";

  import type { Gradio, SelectData } from "@gradio/utils";
  import File from "./shared/File.svelte";
  import FileUpload from "./shared/FileUpload.svelte";
  import type { FileData } from "@gradio/client";
  import { Block, UploadText } from "@gradio/atoms";

  import { StatusTracker } from "@gradio/statustracker";
  import type { LoadingStatus } from "@gradio/statustracker";

  export let elem_id = "";
  export let elem_classes: string[] = [];
  export let visible = true;
  export let value: null | FileData | FileData[];

  export let interactive: boolean;
  export let root: string;
  export let label: string;
  export let show_label: boolean;
  export let height: number | undefined = undefined;

  //Molecule3D specific arguments
  export let reps: any = [];
  export let config: any = {};
  export let confidenceLabel: string = "";
  export let showviewer: boolean = true;

  export let _selectable = false;
  export let loading_status: LoadingStatus;
  export let container = true;
  export let scale: number | null = null;
  export let min_width: number | undefined = undefined;
  export let gradio: Gradio<{
    change: never;
    error: string;
    upload: never;
    clear: never;
    select: SelectData;
    clear_status: LoadingStatus;
    delete: FileData;
  }>;
  export let file_count: "single" | "multiple" | "directory";
  export let file_types: string[] = ["file"];

  let old_value = value;
  $: if (JSON.stringify(old_value) !== JSON.stringify(value)) {
    gradio.dispatch("change");
    old_value = value;

    moldata = null;
    retrieveContent(value);
  }

  $: if (JSON.stringify(reps) !== JSON.stringify(reps)) {
    gradio.dispatch("change");
    old_value = value;

    // console.log("reps changed");
    // console.log(reps);

    moldata = null;
    retrieveContent(value);
  }

  let dragging = false;
  let pending_upload = false;

  let keys_for_reps = {
    model: {
      type: Number,
      default: 0,
    },
    chain: {
      type: String,
      default: "",
    },
    resname: {
      type: String,
      default: "",
    },
    style: {
      type: String,
      default: "cartoon",
      choices: ["cartoon", "stick", "sphere", "surface"],
    },
    color: {
      type: String,
      default: "whiteCarbon",
      choices: [
        "whiteCarbon",
        "orangeCarbon",
        "redCarbon",
        "blackCarbon",
        "blueCarbon",
        "grayCarbon",
        "greenCarbon",
        "cyanCarbon",
        "alphafold",
        "default",
        "Jmol",
        "chain",
        "spectrum",
      ],
    },
    opacity: {
      type: Number,
      default: 1,
    },
    residue_range: {
      type: String,
      default: "",
    },
    around: {
      type: Number,
      default: 0,
    },
    byres: {
      type: Boolean,
      default: false,
    },
    visible: {
      type: Boolean,
      default: true,
    },
  };
  let moldata = null;
  let allowed_extensions = ["pdb", "sdf", "mol2", "pdb1", "cif", "traj", "xyz"];
  async function fetchFileContent(url) {
    const response = await fetch(url);
    const content = await response.text();
    return content;
  }
  let promise = null;
  let errors = [];
  async function retrieveContent(value) {
    if (value == null) {
      return [];
    } else if (Array.isArray(value)) {
      let tempMoldata = [];
      // get file extension
      for (const element of value) {
        let ext = element.orig_name.split(".").pop();
        if (!allowed_extensions.includes(ext)) {
          errors.push(
            `Invalid file extension for ${
              element.orig_name
            }. Expected one of ${allowed_extensions.join(", ")}, got ${ext}`
          );
          moldata = null;
          continue;
        }
        tempMoldata.push({
          data: await fetchFileContent(element.url),
          name: element.orig_name,
          format: ext,
          asFrames: false,
        });
      }
      moldata = tempMoldata;
    } else if (typeof value === "object" && value !== null) {
      let ext = value.orig_name.split(".").pop();
      if (!allowed_extensions.includes(ext)) {
        errors.push(
          `Invalid file extension for ${
            value.orig_name
          }. Expected one of ${allowed_extensions.join(", ")}, got ${ext}`
        );
        moldata = null;
      } else {
        let t = await fetchFileContent(value.url);
        let ext = value.orig_name.split(".").pop();
        if (ext === "pdb1") {
          ext = "pdb";
        }
        moldata = [
          { data: t, name: value.orig_name, format: ext, asFrames: false },
        ];
      }
    } else {
      moldata = null;
    }
    // value is object
  }
  // go through each rep, do a type check and add missing keys to the rep
  let lenMoldata = 0;
  $: lenMoldata = moldata ? moldata.length : 0;
  let representations = [];
  //first add all missing keys
  $: {
    reps.forEach((rep) => {
      for (const [key, value] of Object.entries(keys_for_reps)) {
        if (rep[key] === undefined) {
          rep[key] = value["default"];
        }
        if (rep[key].constructor != value["type"]) {
          errors.push(
            `Invalid type for ${key} in reps. Expected ${
              value["type"]
            }, got ${typeof rep[key]}`
          );
        }
      }
    });
    // then check if model does exist and add to representations
    reps.forEach((rep) => {
      if (rep["model"] <= lenMoldata) {
        representations.push(rep);
      }
    });
  }
  $: promise = retrieveContent(value);
</script>

<Block
  {visible}
  variant={value ? "solid" : "dashed"}
  border_mode={dragging ? "focus" : "base"}
  padding={false}
  {elem_id}
  {elem_classes}
  {container}
  {scale}
  {min_width}
  allow_overflow={false}
>
  <StatusTracker
    autoscroll={gradio.autoscroll}
    i18n={gradio.i18n}
    {...loading_status}
    status={pending_upload
      ? "generating"
      : loading_status?.status || "complete"}
    on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
  />
  {#if !interactive}
    <File
      on:select={({ detail }) => gradio.dispatch("select", detail)}
      selectable={_selectable}
      {value}
      {label}
      {show_label}
      {height}
      {representations}
      {config}
      {confidenceLabel}
      {moldata}
      {errors}
      i18n={gradio.i18n}
      molviewer={showviewer}
    />
  {:else}
    <FileUpload
      upload={gradio.client.upload}
      stream_handler={gradio.client.stream}
      {label}
      {show_label}
      {value}
      {file_count}
      {file_types}
      selectable={_selectable}
      {root}
      {height}
      {representations}
      {config}
      {confidenceLabel}
      {moldata}
      molviewer={showviewer}
      max_file_size={gradio.max_file_size}
      on:change={({ detail }) => {
        value = detail;
      }}
      on:drag={({ detail }) => (dragging = detail)}
      on:clear={() => gradio.dispatch("clear")}
      on:select={({ detail }) => gradio.dispatch("select", detail)}
      on:notfound={() =>
        gradio.dispatch(
          "error",
          "identifier not found in database, check spelling"
        )}
      on:upload={() => gradio.dispatch("upload")}
      on:error={({ detail }) => {
        loading_status = loading_status || {};
        loading_status.status = "error";
        gradio.dispatch("error", detail);
      }}
      on:delete={({ detail }) => {
        gradio.dispatch("delete", detail);
      }}
      i18n={gradio.i18n}
    >
      <UploadText i18n={gradio.i18n} type="file" />
    </FileUpload>
  {/if}

  {#if errors.length > 0 && value !== null}
    <div
      class="flex m-2 p-4 mb-4 text-sm text-red-800 rounded-lg bg-red-50 dark:bg-gray-800 dark:text-red-400"
      role="alert"
    >
      <svg
        class="flex-shrink-0 inline w-4 h-4 mr-3 mt-[2px]"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke-width="1.5"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
        />
      </svg>

      <span class="sr-only">Error in the representations</span>
      <div>
        <span class="font-medium"
          >Couldn't display Molecule. Fix the following problems:</span
        >
        <ul class="mt-1.5 ml-4 list-disc list-inside">
          {#each errors as error}
            <li>{error}</li>
          {/each}
        </ul>
      </div>
    </div>
  {/if}
</Block>
