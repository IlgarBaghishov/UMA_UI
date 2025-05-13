<script lang="ts">
  import { createEventDispatcher, tick } from "svelte";
  import { Upload, ModifyUpload } from "@gradio/upload";
  import type { FileData, Client } from "@gradio/client";
  import { BlockLabel } from "@gradio/atoms";
  import { File } from "@gradio/icons";
  import FileRetrieval from "./FileRetrieval.svelte";

  import FilePreview from "./FilePreview.svelte";
  import type { I18nFormatter } from "@gradio/utils";
  import MolecularViewer from "./MolecularViewer.svelte";

  export let value: null | FileData | FileData[];

  export let label: string;
  export let show_label = true;
  export let file_count: "single" | "multiple" | "directory" = "single";
  export let file_types: string[] | null = null;
  export let selectable = false;
  export let root: string;
  export let height: number | undefined = undefined;
  export let i18n: I18nFormatter;
  export let max_file_size: number | null = null;
  export let upload: Client["upload"];
  export let stream_handler: Client["stream"];

  export let config = {};
  export let confidenceLabel = "";
  export let representations = [];
  export let moldata = null;
  export let molviewer = false;

  async function handle_upload({
    detail,
  }: CustomEvent<FileData | FileData[]>): Promise<void> {
    value = detail;
    console.log("upload", detail);

    await tick();
    dispatch("change", value);
    dispatch("upload", detail);
  }

  function handle_clear(): void {
    value = null;
    oldata = null;
    dispatch("change", null);
    dispatch("clear");
  }

  const dispatch = createEventDispatcher<{
    change: FileData[] | FileData | null;
    clear: undefined;
    drag: boolean;
    upload: FileData[] | FileData;
    load: FileData[] | FileData;
    error: string;
  }>();

  let dragging = false;
  $: dispatch("drag", dragging);
</script>

<BlockLabel {show_label} Icon={File} float={!value} label={label || "File"} />

{#if value && (Array.isArray(value) ? value.length > 0 : true)}
  <ModifyUpload {i18n} on:clear={handle_clear} absolute />
  <FilePreview
    {i18n}
    on:select
    {selectable}
    {value}
    {height}
    on:change
    on:delete
  />
  {#if moldata != null && molviewer}
    <MolecularViewer {moldata} {config} {confidenceLabel} {representations} />
  {/if}
{:else}
  <FileRetrieval
    {root}
    on:load={handle_upload}
    {upload}
    on:notfound={() => dispatch("notfound")}
  />
  <Upload
    on:load={handle_upload}
    filetype={file_types}
    {file_count}
    {max_file_size}
    {root}
    bind:dragging
    on:error
    {stream_handler}
    {upload}
  >
    <slot />
  </Upload>
{/if}
