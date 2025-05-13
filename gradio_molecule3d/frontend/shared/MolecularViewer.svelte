<script lang="ts">
  import * as mol from "3dmol/build/3Dmol.js";
  let TDmol = mol;

  import { onMount, onDestroy, beforeUpdate } from "svelte";

  let viewer;
  export let confidenceLabel = null;

  export let moldata = null;

  let settings = {
    backgroundColor: {
      type: "select",
      options: ["white", "black", "gray", "lightgray", "beige", "orange"],
    },
    orthographic: {
      type: "toggle",
    },
    disableFog: {
      type: "toggle",
    },
  };
  export let config = {
    backgroundColor: "white",
    orthographic: false,
    disableFog: false,
  };

  $: {
    if (view != undefined) {
      view.setBackgroundColor(config.backgroundColor);
      view.enableFog(!config.disableFog);
      view.setCameraParameters({ orthographic: config.orthographic });
    }
  }

  let labelHover = true;

  let showCiteTooltip = false;

  function toggleCiteTooltip() {
    showCiteTooltip = !showCiteTooltip;
  }

  export let representations = [];

  let showOffCanvas = false;
  let showOffCanvasReps = false;

  function toggleOffCanvas() {
    showOffCanvas = !showOffCanvas;
  }

  function toggleOffCanvasReps() {
    showOffCanvasReps = !showOffCanvasReps;
  }
  function deleteRep(index) {
    representations.splice(index, 1);
    representations = representations;
  }
  function insertRep() {
    representations.push({
      model: 0,
      chain: "",
      resname: "",
      style: "cartoon",
      color: "grayCarbon",
      residue_range: "",
      around: 0,
      byres: false,
      visible: true,
      opacity: 1,
      scale: 1,
    });
    representations = representations;
  }

  function fade(node, { delay = 0, duration = 50 }) {
    const o = +getComputedStyle(node).opacity;

    return {
      delay,
      duration,
      css: (t) => `opacity: ${t * o}`,
    };
  }
  let colorAlpha = function (atom) {
    if (atom.b < 50) {
      return "OrangeRed";
    } else if (atom.b < 70) {
      return "Gold";
    } else if (atom.b < 90) {
      return "MediumTurquoise";
    } else {
      return "Blue";
    }
  };

  let colorHydrophobicity = function (atom) {
    let kyte_doolittle = {
      ILE: 4.5,
      VAL: 4.2,
      LEU: 3.8,
      PHE: 2.8,
      CYS: 2.5,
      MET: 1.9,
      ALA: 1.8,
      GLY: -0.4,
      THR: -0.7,
      SER: -0.8,
      TRP: -0.9,
      TYR: -1.3,
      PRO: -1.6,
      HIS: -3.2,
      GLU: -3.5,
      GLN: -3.5,
      ASP: -3.5,
      ASN: -3.5,
      LYS: -3.9,
      ARG: -4.5,
    };
    if (atom.resn in kyte_doolittle) {
      const value = kyte_doolittle[atom.resn];
      const min = -4.5;
      const max = 4.5;

      // Normalize the value to a range of 0 to 1
      const normalized = (value - min) / (max - min);

      // Interpolate colors between DarkCyan, White, and DarkGoldenRod
      const interpolateColor = (start, end, factor) => {
        const startRGB = parseInt(start.slice(1), 16);
        const endRGB = parseInt(end.slice(1), 16);

        const r = Math.round(
          ((endRGB >> 16) - (startRGB >> 16)) * factor + (startRGB >> 16),
        );
        const g = Math.round(
          (((endRGB >> 8) & 0xff) - ((startRGB >> 8) & 0xff)) * factor +
            ((startRGB >> 8) & 0xff),
        );
        const b = Math.round(
          ((endRGB & 0xff) - (startRGB & 0xff)) * factor + (startRGB & 0xff),
        );

        return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
      };

      if (normalized <= 0.5) {
        // Interpolate between DarkCyan and White
        return interpolateColor("#008B8B", "#FFFFFF", normalized * 2);
      } else {
        // Interpolate between White and DarkGoldenRod
        return interpolateColor("#FFFFFF", "#B8860B", (normalized - 0.5) * 2);
      }
    } else {
      return "#FFFFFF"; // White
    }
  };

  let view;

  function resetZoom(rep) {
    // if is not pointerevent
    if (rep.type != undefined) {
      view.zoomTo();
    } else {
      let sel = {
        model: rep.model,
      };
      if (rep.chain !== "") {
        sel.chain = rep.chain;
      }
      if (rep.residue_range !== "") {
        sel.resi = rep.residue_range;
      }
      if (rep.resname !== "") {
        sel.resn = rep.resname;
      }
      view.zoomTo(sel);
    }
  }
  function applyStyles(representations) {
    if (view !== undefined) {
      view.setStyle();
      view.removeAllSurfaces();
      representations.forEach((rep) => {
        let colorObj;

        if (rep.color === "alphafold") {
          colorObj = { colorfunc: colorAlpha };
        } else if (rep.color === "hydrophobicity") {
          colorObj = { colorfunc: colorHydrophobicity };
        } else if (rep.color == "spectrum") {
          colorObj = { color: "spectrum" };
        } else {
          colorObj = { colorscheme: rep.color };
        }

        colorObj.opacity = rep.opacity;

        let selObj = { model: rep.model };
        if (rep.chain !== "") {
          selObj.chain = rep.chain;
        }
        if (rep.residue_range !== "") {
          selObj.resi = rep.residue_range;
        }
        if (rep.resname !== "") {
          selObj.resn = rep.resname;
        }
        selObj.byres = rep.byres;
        if (rep.around !== 0) {
          selObj.expand = rep.around;
        }
        if (rep.sidechain) {
          selObj = {
            and: [selObj, { atom: ["N", "C", "O"], invert: true }],
          };
        }

        if (rep.style == "sphere") {
          colorObj["scale"] = rep.scale;
        } else if (rep.style == "stick") {
          colorObj["scale"] = rep.scale;
        }

        if (rep.style === "surface") {
          view.addSurface(TDmol.SurfaceType.VDW, colorObj, selObj);
        } else {
          try {
            if (view.getModel(selObj.model) != null) {
              view.addStyle(selObj, {
                [rep.style]: colorObj,
              });
            }
          } catch (error) {
            console.log(error);
          }
        }
      });

      view.render();
    }
  }

  onMount(() => {
    console.log("MolecularViewer Mounted");
    let startingConfig = { ...config, cartoonQuality: 7 };

    view = TDmol.createViewer(viewer, startingConfig);

    //filter duplicate representations
    let uniqueReps = [];
    representations.forEach((rep) => {
      if (
        !uniqueReps.some(
          (uniqueRep) =>
            uniqueRep.model === rep.model &&
            uniqueRep.chain === rep.chain &&
            uniqueRep.resname === rep.resname &&
            uniqueRep.style === rep.style &&
            uniqueRep.color === rep.color &&
            uniqueRep.residue_range === rep.residue_range &&
            uniqueRep.around === rep.around &&
            uniqueRep.byres === rep.byres &&
            uniqueRep.sidechain === rep.sidechain &&
            uniqueRep.scale == rep.scale,
        )
      ) {
        uniqueReps.push(rep);
      }
    });
    representations = uniqueReps;
    if (moldata.length > 0) {
      moldata.forEach((element) => {
        view.addModelsAsFrames(element.data, element.format, {
          keepH: true,
          multiModel: true,
        });
      });
    }

    view.addUnitCell(view.getModel(), {
      box: {
        color: "black", // Set color to black
        linewidth: 2, // Set line width
        hidden: false,
      },
    });

    applyStyles(representations);
    view.zoomTo();
    view.render();
    view.zoom(0.9, 100);

    representations.forEach((rep) => {
      if (rep.color === "alphafold") {
        anyColorAlphaFold = true;
      }
      if (rep.color === "hydrophobicity") {
        anyColorHydrophobic = true;
      }
    });

    if (labelHover) {
      view.setHoverable(
        {},
        true,
        function (atom, view, event, container) {
          if (!atom.label) {
            let label;
            if (anyColorAlphaFold) {
              label =
                atom.resn +
                ":" +
                atom.resi +
                ":" +
                atom.atom +
                " (" +
                confidenceLabel +
                " " +
                atom.b +
                ")";
            } else {
              label = atom.resn + ":" + atom.resi + ":" + atom.atom;
            }
            atom.label = view.addLabel(label, {
              position: atom,
              backgroundColor: "#ffffff",
              borderColor: "#dddddd",
              fontColor: "black",
            });
          }
        },
        function (atom, view) {
          if (atom.label) {
            view.removeLabel(atom.label);
            delete atom.label;
          }
        },
      );
    }

    if (view.getNumFrames() > 1) {
      toggleAnimation();
    }
  });

  beforeUpdate(() => {
    // console.log("beforeUpdate");
    // console.log(representations);
  });

  $: applyStyles(representations);
  let hasFrames = true;
  let isAnimated = false;

  let anyColorAlphaFold = false;
  $: {
    anyColorAlphaFold = false;
    representations.forEach((rep) => {
      if (rep.color === "alphafold") {
        anyColorAlphaFold = true;
      }
    });
  }

  let anyColorHydrophobic = false;
  $: {
    anyColorHydrophobic = false;
    representations.forEach((rep) => {
      if (rep.color === "hydrophobicity") {
        anyColorHydrophobic = true;
      }
    });
  }
  function toggleAnimation() {
    console.log(view.isAnimated());
    if (isAnimated) {
      view.stopAnimate();
    } else {
      view.animate({ loop: "forward", reps: 0 });
    }
    isAnimated = !isAnimated;
  }
</script>

<div class="bg-white w-full minh">
  <div class="overflow-hidden flex gap-px w-full h-full flex-wrap">
    <div class="gr-block gr-box relative w-full overflow-hidden">
      <div
        class="absolute z-50 top-0 right-0 mr-2 flex flex-col divide-y border border-gray-200 mt-2 rounded items-center justify-center bg-white dark:bg-gray-800"
      >
        <button class="p-2" title="Reset View" on:click={resetZoom}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
            class="w-4 h-4 text-gray-500 hover:text-orange-600 cursor-pointer"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25"
            />
          </svg>
        </button>
        <button class="p-2" title="Settings" on:click={toggleOffCanvas}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
            class="w-5 h-5 text-gray-500 hover:text-orange-600 cursor-pointer"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75"
            />
          </svg>
        </button>

        <button
          class="p-2"
          title="Representations"
          on:click={toggleOffCanvasReps}
        >
          <!-- <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
            class="w-5 h-5 text-gray-500 hover:text-orange-600 cursor-pointer"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M21.75 6.75a4.5 4.5 0 01-4.884 4.484c-1.076-.091-2.264.071-2.95.904l-7.152 8.684a2.548 2.548 0 11-3.586-3.586l8.684-7.152c.833-.686.995-1.874.904-2.95a4.5 4.5 0 016.336-4.486l-3.276 3.276a3.004 3.004 0 002.25 2.25l3.276-3.276c.256.565.398 1.192.398 1.852z"
            />
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M4.867 19.125h.008v.008h-.008v-.008z"
            />
          </svg> -->

          <!-- CC BY ND  zwicon https://www.zwicon.com -->
          <svg
            class="w-5 h-5 text-gray-500 hover:text-orange-600 cursor-pointer"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M12.0281146,15 L12.5,15 C13.3284271,15 14,15.6715729 14,16.5 L14,20.5 C14,21.3284271 13.3284271,22 12.5,22 L10.5,22 C9.67157288,22 9,21.3284271 9,20.5 L9,16.5 C9,15.6715729 9.67157288,15 10.5,15 L11.0167145,15 C11.1492163,13.8570393 12.0552513,12.9352285 13.2239212,12.8053762 L18.6656473,12.20074 C19.4252963,12.1163346 20,11.4742382 20,10.7099144 L20,8.5 C20,7.67157288 19.3284271,7 18.5,7 C18.2238576,7 18,6.77614237 18,6.5 C18,6.22385763 18.2238576,6 18.5,6 C19.8807119,6 21,7.11928813 21,8.5 L21,10.7099144 C21,11.9837875 20.0421605,13.053948 18.7760788,13.1946238 L13.3343527,13.79926 C12.6731623,13.8727256 12.1520824,14.3686853 12.0281146,15 Z M10.5,16 C10.2238576,16 10,16.2238576 10,16.5 L10,20.5 C10,20.7761424 10.2238576,21 10.5,21 L12.5,21 C12.7761424,21 13,20.7761424 13,20.5 L13,16.5 C13,16.2238576 12.7761424,16 12.5,16 L10.5,16 Z M5.49996942,2.99995128 L16.5,2.99995128 C17.8807119,2.99995128 19,4.11923941 19,5.49995128 L19,7.5 C19,8.88071187 17.8807119,10 16.5,10 L5.49996942,10 C4.11925755,10 2.99996942,8.88071187 2.99996942,7.5 L2.99996942,5.49995128 C2.99996942,4.11923941 4.11925755,2.99995128 5.49996942,2.99995128 Z M5.49996942,3.99995128 C4.6715423,3.99995128 3.99996942,4.67152416 3.99996942,5.49995128 L3.99996942,7.5 C3.99996942,8.32842712 4.6715423,9 5.49996942,9 L16.5,9 C17.3284271,9 18,8.32842712 18,7.5 L18,5.49995128 C18,4.67152416 17.3284271,3.99995128 16.5,3.99995128 L5.49996942,3.99995128 Z"
            />
          </svg>
        </button>

        <!-- <button class="p-2" title="Download files">
          <svg
            aria-hidden="true"
            fill="currentColor"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
            class="w-5 h-5 text-gray-500 hover:text-orange-600 cursor-pointer"
          >
            <path
              d="M10.75 2.75a.75.75 0 00-1.5 0v8.614L6.295 8.235a.75.75 0 10-1.09 1.03l4.25 4.5a.75.75 0 001.09 0l4.25-4.5a.75.75 0 00-1.09-1.03l-2.955 3.129V2.75z"
            />
            <path
              d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z"
            />
          </svg>
        </button> -->
      </div>

      {#if anyColorAlphaFold}
        <div class="absolute bottom-0 left-0 z-50 p-2 -mb-2 w-full bg-white">
          <div class="flex text-sm items-center space-x-2 justify-center">
            <div class="flex space-x-1 py-1 items-center">
              <span class="w-4 h-4" style="background-color: rgb(0, 83, 214);"
                >&nbsp;</span
              ><span class="legendlabel"
                >Very high ({confidenceLabel} &gt; 90)</span
              >
            </div>
            <div class="flex space-x-1 py-1 items-center">
              <span
                class="w-4 h-4"
                style="background-color: rgb(101, 203, 243);">&nbsp;</span
              ><span class="legendlabel"
                >Confident (90 &gt; {confidenceLabel} &gt; 70)</span
              >
            </div>
            <div class="flex space-x-1 py-1 items-center">
              <span class="w-4 h-4" style="background-color: rgb(255, 219, 19);"
                >&nbsp;</span
              ><span class="legendlabel"
                >Low (70 &gt; {confidenceLabel} &gt; 50)</span
              >
            </div>
            <div class="flex space-x-1 py-1 items-center">
              <span class="w-4 h-4" style="background-color: rgb(255, 125, 69);"
                >&nbsp;</span
              ><span class="legendlabel"
                >Very low ({confidenceLabel} &lt; 50)</span
              >
            </div>
          </div>
        </div>
      {/if}

      {#if anyColorHydrophobic}
        <div class="absolute bottom-0 left-0 z-50 p-2 mb-2 w-full bg-white">
          <div class="flex text-sm items-center space-x-2 justify-center">
            <div class="flex space-x-1 py-1 items-center">
              <a
                class="cursor-pointer"
                href="https://pubmed.ncbi.nlm.nih.gov/7108955/"
                >Kyte & Doolittle hydrophobicity scale:</a
              >
            </div>
            <div class="flex space-x-1 py-1 items-center">
              <span class="w-4 h-4" style="background-color: #B8860B;"
                >&nbsp;</span
              ><span class="legendlabel">Hydrophobic</span>
            </div>
            <div class="flex space-x-1 py-1 items-center">
              <span
                class="w-4 h-4 border border-gray-300"
                style="background-color: white">&nbsp;</span
              ><span class="legendlabel">Neutral</span>
            </div>
            <div class="flex space-x-1 py-1 items-center">
              <span class="w-4 h-4" style="background-color: darkcyan;"
                >&nbsp;</span
              ><span class="legendlabel">Hydrophilic</span>
            </div>
          </div>
        </div>
      {/if}

      {#if hasFrames}
        <div
          class="absolute z-50 bottom-0 right-0 mr-2 flex divide-x border border-gray-200 mb-2 rounded items-center justify-center"
        >
          {#if !isAnimated}
            <button class="p-2" title="Play" on:click={toggleAnimation}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke-width="2"
                stroke="currentColor"
                class="w-4 h-4 text-gray-500 hover:text-orange-600 cursor-pointer"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z"
                />
              </svg>
            </button>
          {:else}
            <button class="p-2" title="Pause" on:click={toggleAnimation}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke-width="2"
                stroke="currentColor"
                class="w-4 h-4 text-gray-500 hover:text-orange-600 cursor-pointer"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M5.25 7.5A2.25 2.25 0 017.5 5.25h9a2.25 2.25 0 012.25 2.25v9a2.25 2.25 0 01-2.25 2.25h-9a2.25 2.25 0 01-2.25-2.25v-9z"
                />
              </svg>

              <!-- <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke-width="2"
                                stroke="currentColor"
                                class="w-4 h-4 text-gray-500 hover:text-orange-600 cursor-pointer"
                            >
                                <path
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    d="M15.75 5.25v13.5m-7.5-13.5v13.5"
                                />
                            </svg> -->
            </button>
          {/if}
        </div>
      {/if}

      <div
        class="absolute z-50 bottom-0 left-0 ml-2 flex divide-x mb-2 rounded items-center justify-center"
      >
        <button
          class="p-1"
          title="How to cite 3Dmol.js"
          on:click={toggleCiteTooltip}
        >
          <svg
            class="w-4 h-4 text-gray-500 hover:text-orange-600 cursor-pointer fill-current"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
            ><g id="SVGRepo_bgCarrier" stroke-width="0" /><g
              id="SVGRepo_tracerCarrier"
              stroke-linecap="round"
              stroke-linejoin="round"
            /><g id="SVGRepo_iconCarrier"
              ><path
                d="M12.6 7.6v3.9h-1.1v4.6h-3v-4.6H7.4V7.6c0-.3.3-.6.6-.6h4c.3 0 .6.3.6.6zM10 6.5c.7 0 1.3-.6 1.3-1.3 0-.7-.6-1.3-1.3-1.3-.7 0-1.3.6-1.3 1.3 0 .7.6 1.3 1.3 1.3zm9.6 3.5c0 2.7-.9 4.9-2.7 6.7-1.9 1.9-4.2 2.9-6.9 2.9-2.6 0-4.9-.9-6.8-2.8C1.3 14.9.4 12.7.4 10c0-2.6.9-4.9 2.8-6.8C5.1 1.3 7.3.4 10 .4s5 .9 6.8 2.8c1.9 1.8 2.8 4.1 2.8 6.8zm-1.7 0c0-2.2-.8-4-2.3-5.6C14 2.9 12.2 2.1 10 2.1c-2.2 0-4 .8-5.5 2.3C2.9 6 2.1 7.9 2.1 10c0 2.1.8 4 2.3 5.5s3.4 2.3 5.6 2.3c2.1 0 4-.8 5.6-2.4 1.5-1.4 2.3-3.2 2.3-5.4z"
              /></g
            ></svg
          >
        </button>

        <span
          class=" absolute -top-1 left-6 w-max rounded bg-gray-900 px-2 py-1 text-sm font-medium text-gray-50 opacity-0 shadow transition-opacity"
          class:opacity-100={showCiteTooltip}
        >
          Cite 3dmol.js as Rego & Koes, doi:10/gb5g5n
        </span>
      </div>

      <div class="viewer w-full h-full z-10" bind:this={viewer} />

      {#if showOffCanvas}
        <div
          id="settings-drawer"
          class="absolute top-0 right-0 z-50 h-full overflow-y-auto transition-transform bg-gray-100 w-80 dark:bg-gray-800"
          tabindex="-1"
          aria-labelledby="settings-drawer-label"
        >
          <div class="p-4">
            <h5
              id="settings-drawer-label"
              class="inline-flex items-center mb-4 text-base font-semibold text-gray-500 dark:text-gray-400"
            >
              Settings
            </h5>
            <button
              type="button"
              data-drawer-hide="drawer-example"
              aria-controls="drawer-example"
              class="text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm p-1.5 absolute top-2.5 right-2.5 inline-flex items-center dark:hover:bg-gray-600 dark:hover:text-white"
              on:click={toggleOffCanvas}
            >
              <svg
                aria-hidden="true"
                class="w-5 h-5"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
                ><path
                  fill-rule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                  clip-rule="evenodd"
                /></svg
              >
              <span class="sr-only">Close menu</span>
            </button>
          </div>
          {#each Object.keys(config) as setting}
            <div class="flex flex-col mb-4 divide-y">
              <div
                class="flex items-center border-t border-b border-gray-200 bg-white px-4 py-2 space-x-2"
              >
                <label
                  for={setting}
                  class="text-sm font-medium text-gray-600 dark:text-gray-400 w-1/2"
                >
                  {setting}
                </label>

                {#if settings[setting].type == "toggle"}
                  <label
                    class="relative inline-flex items-center mr-5 cursor-pointer text-center justify-center"
                  >
                    <input
                      type="checkbox"
                      value=""
                      class="sr-only peer"
                      bind:checked={config[setting]}
                    />
                    <div
                      class="w-11 h-6 bg-gray-200 rounded-full peer dark:bg-gray-700 peer-focus:ring-4 peer-focus:ring-orange-300 dark:peer-focus:ring-orange-800 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-orange-400"
                    />
                  </label>
                {/if}
                {#if settings[setting].type == "range"}
                  <div class="flex items-center">
                    <input
                      id="medium-range"
                      type="range"
                      value={config[setting]}
                      min={settings[setting].min}
                      max={settings[setting].max}
                      step={settings[setting].step}
                      on:change={() => {
                        config[setting] = event.target.value;
                      }}
                      class="w-2/3 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    />
                    <span
                      class="w-1/3 text-center text-sm font-medium text-gray-600 dark:text-gray-400"
                      >{config[setting]}</span
                    >
                  </div>
                {/if}
                {#if settings[setting].type == "select"}
                  <label for={setting} class="sr-only">Select style</label>
                  <select
                    id={setting}
                    class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                    bind:value={config[setting]}
                  >
                    {#each settings[setting].options as option}
                      <option
                        value={option}
                        selected={option == config[setting]}
                      >
                        {option}
                      </option>
                    {/each}
                  </select>
                {/if}
              </div>
            </div>
          {/each}
          <div class="bg-white">
            <div
              class="flex items-center border-t border-b border-gray-200 bg-white px-4 py-2 space-x-2"
            >
              <label
                class="text-sm font-medium text-gray-600 dark:text-gray-400 w-1/2"
              >
                Label atoms on hover
              </label>
              <label
                class="relative inline-flex items-center mr-5 cursor-pointer text-center justify-center"
              >
                <input
                  type="checkbox"
                  class="sr-only peer"
                  bind:value={labelHover}
                />
                <div
                  class="w-11 h-6 bg-gray-200 rounded-full peer dark:bg-gray-700 peer-focus:ring-4 peer-focus:ring-orange-300 dark:peer-focus:ring-orange-800 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-orange-400"
                />
              </label>
            </div>
          </div>
        </div>
      {/if}
      {#if showOffCanvasReps}
        <div
          id="drawer-example"
          class="absolute top-0 right-0 z-50 h-full overflow-y-auto transition-transform bg-gray-100 w-80 dark:bg-gray-800 border-l border-gray-200"
          tabindex="-1"
          aria-labelledby="drawer-label"
        >
          <div class="p-4">
            <h5
              id="drawer-label"
              class="inline-flex items-center mb-4 text-base font-semibold text-gray-500 dark:text-gray-400"
            >
              Representations
            </h5>
            <button
              type="button"
              data-drawer-hide="drawer-example"
              aria-controls="drawer-example"
              class="text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm p-1.5 absolute top-2.5 right-2.5 inline-flex items-center dark:hover:bg-gray-600 dark:hover:text-white"
              on:click={toggleOffCanvasReps}
            >
              <svg
                aria-hidden="true"
                class="w-5 h-5"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
                ><path
                  fill-rule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                  clip-rule="evenodd"
                /></svg
              >
              <span class="sr-only">Close menu</span>
            </button>
          </div>

          {#each representations as rep, index}
            <div class="bg-white border-b border-t border-gray-200 py-4 px-2">
              <div class="">
                <div class="flex space-x-2 items-center cursor-pointer p-1">
                  <button
                    on:click={() => (rep.visible = !rep.visible)}
                    class="flex items-center space-x-2"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke-width="1.5"
                      stroke="currentColor"
                      class={rep.visible
                        ? "transform rotate-90 w-5 h-5 "
                        : "w-5 h-5 "}
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M8.25 4.5l7.5 7.5-7.5 7.5"
                      />
                    </svg>
                    <span>Representation #{index}</span>
                  </button>
                  <button on:click={() => deleteRep(index)}>
                    <!---->
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke-width="1.5"
                      stroke="currentColor"
                      class="w-5 h-5"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0"
                      />
                    </svg>
                  </button>
                  <button
                    title="zoom to selection"
                    on:click={() => resetZoom(rep)}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke-width="1.5"
                      stroke="currentColor"
                      class="w-4 h-4 text-gray-500 hover:text-orange-600 cursor-pointer"
                    >
                      <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25"
                      />
                    </svg>
                  </button>
                </div>
                {#if rep.visible}
                  <div in:fade>
                    <div class="p-1 flex space-x-1">
                      <select
                        id="style"
                        class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                        bind:value={rep.model}
                      >
                        {#each moldata as mol, i}
                          <option value={i}>{mol.name} #{i}</option>
                        {/each}
                      </select>
                      <input
                        type="text"
                        id="chain"
                        class="w-1/2 bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                        placeholder="Chain"
                        bind:value={rep.chain}
                      />
                    </div>
                    <div class="p-1 flex space-x-1">
                      <input
                        type="text"
                        id="chain"
                        class="w-1/2 bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                        placeholder="Resname"
                        bind:value={rep.resname}
                      />
                      <input
                        type="text"
                        id="residue_range"
                        class="w-full bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                        placeholder="Residue range"
                        bind:value={rep.residue_range}
                      />
                    </div>
                    <div class="p-1 flex space-x-1 items-center">
                      <label
                        for="countries"
                        class="block mb-2 text-sm w-1/3 font-medium text-gray-600 dark:text-white"
                        >Select style</label
                      >
                      <select
                        id="style"
                        class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                        bind:value={rep.style}
                      >
                        <option value="stick">Stick</option>
                        <option value="cartoon">Cartoon</option>
                        <option value="surface">Surface</option>$
                        <option value="sphere">Sphere</option>
                      </select>
                    </div>
                    <div
                      class="flex p-1 items-center text-gray-700 space-x-1 text-sm"
                    >
                      <div class="">Opacity</div>
                      <input
                        id="around"
                        type="range"
                        value={rep.opacity}
                        min="0"
                        max="1"
                        step="0.1"
                        on:change={(event) => {
                          rep.opacity = event.target.value;
                        }}
                        class="h-2 w-full bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                      />
                      <input type="text" bind:value={rep.opacity} class="w-8" />
                    </div>
                    <div
                      class="flex p-1 items-center text-gray-700 space-x-1 text-sm"
                    >
                      <div class="">scale</div>
                      <input
                        id="around"
                        type="range"
                        value={rep.scale}
                        min="0"
                        max="1"
                        step="0.1"
                        on:change={(event) => {
                          rep.scale = event.target.value;
                        }}
                        class="h-2 w-full bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                      />
                      <input type="text" bind:value={rep.scale} class="w-8" />
                    </div>
                    <div
                      class="flex p-1 items-center text-gray-700 space-x-1 text-sm"
                    >
                      <div class="">Expand selection</div>
                      <input
                        id="around"
                        type="range"
                        value={rep.around}
                        min="0"
                        max="10"
                        step="0.5"
                        on:change={(event) => {
                          rep.around = event.target.value;
                        }}
                        class="h-2 w-1/3 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                      />
                      <input type="text" bind:value={rep.around} class="w-8" />
                      <div>Å</div>
                    </div>

                    <div
                      class="flex p-1 items-center text-gray-700 space-x-1 text-sm"
                    >
                      <div class="flex space-x-1 w-1/2">
                        <span>Full residue</span>
                        <label
                          class="relative inline-flex items-center mr-5 cursor-pointer text-center"
                        >
                          <input
                            type="checkbox"
                            value=""
                            class="sr-only peer"
                            bind:checked={rep.byres}
                          />
                          <div
                            class="w-11 h-6 bg-gray-200 rounded-full peer dark:bg-gray-700 peer-focus:ring-4 peer-focus:ring-orange-300 dark:peer-focus:ring-orange-800 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-orange-400"
                          />
                        </label>
                      </div>
                      <div class="flex space-x-1 w-1/2">
                        <span>Only sidechain</span>
                        <label
                          class="relative inline-flex items-center mr-5 cursor-pointer text-center"
                        >
                          <input
                            type="checkbox"
                            value=""
                            class="sr-only peer"
                            bind:checked={rep.sidechain}
                          />
                          <div
                            class="w-11 h-6 bg-gray-200 rounded-full peer dark:bg-gray-700 peer-focus:ring-4 peer-focus:ring-orange-300 dark:peer-focus:ring-orange-800 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-orange-400"
                          />
                        </label>
                      </div>
                    </div>

                    <div class="flex space-x-2 mt-2">
                      <button
                        class={rep.color === "orangeCarbon"
                          ? "bg-orange-600 rounded-full w-8 h-8 border-4 border-gray-300  cursor-pointer"
                          : "bg-orange-600 rounded-full w-8 h-8 border-4 border-white cursor-pointer"}
                        on:click={() => (rep.color = "orangeCarbon")}
                      />
                      <button
                        class={rep.color === "redCarbon"
                          ? "bg-red-600 rounded-full w-8 h-8 border-4 border-gray-300  cursor-pointer"
                          : "bg-red-600 rounded-full w-8 h-8 border-4 border-white cursor-pointer"}
                        on:click={() => (rep.color = "redCarbon")}
                      />
                      <button
                        class={rep.color === "blackCarbon"
                          ? "bg-black rounded-full w-8 h-8 border-4 border-gray-300  cursor-pointer"
                          : "bg-black rounded-full w-8 h-8 border-4 border-white cursor-pointer"}
                        on:click={() => (rep.color = "blackCarbon")}
                      />
                      <button
                        class={rep.color === "blueCarbon"
                          ? "bg-blue-600 rounded-full w-8 h-8 border-4 border-gray-300  cursor-pointer"
                          : "bg-blue-600 rounded-full w-8 h-8 border-4 border-white cursor-pointer"}
                        on:click={() => (rep.color = "blueCarbon")}
                      />
                      <button
                        class={rep.color === "grayCarbon"
                          ? "bg-gray-600 rounded-full w-8 h-8 border-4 border-gray-300  cursor-pointer"
                          : "bg-gray-600 rounded-full w-8 h-8 border-4 border-white cursor-pointer"}
                        on:click={() => (rep.color = "grayCarbon")}
                      />
                      <button
                        class={rep.color === "greenCarbon"
                          ? "bg-green-600 rounded-full w-8 h-8 border-4 border-gray-300  cursor-pointer"
                          : "bg-green-600 rounded-full w-8 h-8 border-4 border-white cursor-pointer"}
                        on:click={() => (rep.color = "greenCarbon")}
                      />
                      <button
                        class={rep.color === "cyanCarbon"
                          ? "bg-cyan-600 rounded-full w-8 h-8 border-4 border-gray-300  cursor-pointer"
                          : "bg-cyan-600 rounded-full w-8 h-8 border-4 border-white cursor-pointer"}
                        on:click={() => (rep.color = "cyanCarbon")}
                      />
                    </div>
                    <div class="flex space-x-2 py-2 text-sm">
                      <button
                        class={rep.color === "alphafold"
                          ? "rounded-lg p-1 border border-gray-400 cursor-pointer bg-gray-200"
                          : "rounded-lg p-1 border border-gray-200 cursor-pointer bg-white"}
                        on:click={() => (rep.color = "alphafold")}
                      >
                        AlphaFold
                      </button>
                      <button
                        class={rep.color === "hydrophobicity"
                          ? "rounded-lg p-1 border border-gray-400 cursor-pointer bg-gray-200"
                          : "rounded-lg p-1 border border-gray-200 cursor-pointer bg-white"}
                        on:click={() => (rep.color = "hydrophobicity")}
                      >
                        Hydrophobicity
                      </button>
                      <button
                        class={rep.color === "default"
                          ? "rounded-lg p-1 border border-gray-400 cursor-pointer bg-gray-200"
                          : "rounded-lg p-1 border border-gray-200 cursor-pointer bg-white"}
                        on:click={() => (rep.color = "default")}
                      >
                        PyMol
                      </button>
                      <button
                        class={rep.color === "Jmol"
                          ? "rounded-lg p-1 border border-gray-400 cursor-pointer bg-gray-200"
                          : "rounded-lg p-1 border border-gray-200 cursor-pointer bg-white"}
                        on:click={() => (rep.color = "Jmol")}
                      >
                        Jmol
                      </button>
                    </div>
                    <div class="flex space-x-2 py-2 text-sm">
                      <button
                        class={rep.color === "chain"
                          ? "rounded-lg p-1 border border-gray-400 cursor-pointer bg-gray-200"
                          : "rounded-lg p-1 border border-gray-200 cursor-pointer bg-white"}
                        on:click={() => (rep.color = "chain")}
                      >
                        Chain
                      </button>
                      <button
                        class={rep.color === "spectrum"
                          ? "rounded-lg p-1 border border-gray-400 cursor-pointer bg-gray-200"
                          : "rounded-lg p-1 border border-gray-200 cursor-pointer bg-white"}
                        on:click={() => (rep.color = "spectrum")}
                      >
                        Spectrum
                      </button>
                    </div>
                  </div>
                {/if}
              </div>
            </div>
          {/each}

          <button
            class="w-full flex text-orange-600 justify-center my-2 text-sm space-x-2 items-center hover:text-gray-600 cursor-pointer"
            on:click={insertRep}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke-width="2"
              stroke="currentColor"
              class="w-4 h-4"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M12 9v6m3-3H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>

            <div>Add representation</div>
          </button>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  *,
  :before,
  :after {
    box-sizing: border-box;
    border-width: 0;
    border-style: solid;
    border-color: #e5e7eb;
  }
  :before,
  :after {
    --tw-content: "";
  }
  html {
    line-height: 1.5;
    -webkit-text-size-adjust: 100%;
    -moz-tab-size: 4;
    -o-tab-size: 4;
    tab-size: 4;
    font-family:
      ui-sans-serif,
      system-ui,
      -apple-system,
      BlinkMacSystemFont,
      Segoe UI,
      Roboto,
      Helvetica Neue,
      Arial,
      Noto Sans,
      sans-serif,
      "Apple Color Emoji",
      "Segoe UI Emoji",
      Segoe UI Symbol,
      "Noto Color Emoji";
    font-feature-settings: normal;
  }
  body {
    margin: 0;
    line-height: inherit;
  }
  hr {
    height: 0;
    color: inherit;
    border-top-width: 1px;
  }
  abbr:where([title]) {
    -webkit-text-decoration: underline dotted;
    text-decoration: underline dotted;
  }
  h1,
  h2,
  h3,
  h4,
  h5,
  h6 {
    font-size: inherit;
    font-weight: inherit;
  }
  a {
    color: inherit;
    text-decoration: inherit;
  }
  b,
  strong {
    font-weight: bolder;
  }
  code,
  kbd,
  samp,
  pre {
    font-family:
      ui-monospace,
      SFMono-Regular,
      Menlo,
      Monaco,
      Consolas,
      Liberation Mono,
      Courier New,
      monospace;
    font-size: 1em;
  }
  small {
    font-size: 80%;
  }
  sub,
  sup {
    font-size: 75%;
    line-height: 0;
    position: relative;
    vertical-align: baseline;
  }
  sub {
    bottom: -0.25em;
  }
  sup {
    top: -0.5em;
  }
  table {
    text-indent: 0;
    border-color: inherit;
    border-collapse: collapse;
  }
  button,
  input,
  optgroup,
  select,
  textarea {
    font-family: inherit;
    font-size: 100%;
    font-weight: inherit;
    line-height: inherit;
    color: inherit;
    margin: 0;
    padding: 0;
  }
  button,
  select {
    text-transform: none;
  }
  button,
  [type="button"],
  [type="reset"],
  [type="submit"] {
    -webkit-appearance: button;
    background-color: transparent;
    background-image: none;
  }
  :-moz-focusring {
    outline: auto;
  }
  :-moz-ui-invalid {
    box-shadow: none;
  }
  progress {
    vertical-align: baseline;
  }
  ::-webkit-inner-spin-button,
  ::-webkit-outer-spin-button {
    height: auto;
  }
  [type="search"] {
    -webkit-appearance: textfield;
    outline-offset: -2px;
  }
  ::-webkit-search-decoration {
    -webkit-appearance: none;
  }
  ::-webkit-file-upload-button {
    -webkit-appearance: button;
    font: inherit;
  }
  summary {
    display: list-item;
  }
  blockquote,
  dl,
  dd,
  h1,
  h2,
  h3,
  h4,
  h5,
  h6,
  hr,
  figure,
  p,
  pre {
    margin: 0;
  }
  fieldset {
    margin: 0;
    padding: 0;
  }
  legend {
    padding: 0;
  }
  ol,
  ul,
  menu {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  textarea {
    resize: vertical;
  }
  input::-moz-placeholder,
  textarea::-moz-placeholder {
    opacity: 1;
    color: #9ca3af;
  }
  input::placeholder,
  textarea::placeholder {
    opacity: 1;
    color: #9ca3af;
  }
  button,
  [role="button"] {
    cursor: pointer;
  }
  :disabled {
    cursor: default;
  }
  img,
  svg,
  video,
  canvas,
  audio,
  iframe,
  embed,
  object {
    display: block;
    vertical-align: middle;
  }
  img,
  video {
    max-width: 100%;
    height: auto;
  }
  [hidden] {
    display: none;
  }
  *,
  :before,
  :after {
    --tw-border-spacing-x: 0;
    --tw-border-spacing-y: 0;
    --tw-translate-x: 0;
    --tw-translate-y: 0;
    --tw-rotate: 0;
    --tw-skew-x: 0;
    --tw-skew-y: 0;
    --tw-scale-x: 1;
    --tw-scale-y: 1;
    --tw-pan-x: ;
    --tw-pan-y: ;
    --tw-pinch-zoom: ;
    --tw-scroll-snap-strictness: proximity;
    --tw-ordinal: ;
    --tw-slashed-zero: ;
    --tw-numeric-figure: ;
    --tw-numeric-spacing: ;
    --tw-numeric-fraction: ;
    --tw-ring-inset: ;
    --tw-ring-offset-width: 0px;
    --tw-ring-offset-color: #fff;
    --tw-ring-color: rgb(59 130 246 / 0.5);
    --tw-ring-offset-shadow: 0 0 #0000;
    --tw-ring-shadow: 0 0 #0000;
    --tw-shadow: 0 0 #0000;
    --tw-shadow-colored: 0 0 #0000;
    --tw-blur: ;
    --tw-brightness: ;
    --tw-contrast: ;
    --tw-grayscale: ;
    --tw-hue-rotate: ;
    --tw-invert: ;
    --tw-saturate: ;
    --tw-sepia: ;
    --tw-drop-shadow: ;
    --tw-backdrop-blur: ;
    --tw-backdrop-brightness: ;
    --tw-backdrop-contrast: ;
    --tw-backdrop-grayscale: ;
    --tw-backdrop-hue-rotate: ;
    --tw-backdrop-invert: ;
    --tw-backdrop-opacity: ;
    --tw-backdrop-saturate: ;
    --tw-backdrop-sepia: ;
  }
  ::backdrop {
    --tw-border-spacing-x: 0;
    --tw-border-spacing-y: 0;
    --tw-translate-x: 0;
    --tw-translate-y: 0;
    --tw-rotate: 0;
    --tw-skew-x: 0;
    --tw-skew-y: 0;
    --tw-scale-x: 1;
    --tw-scale-y: 1;
    --tw-pan-x: ;
    --tw-pan-y: ;
    --tw-pinch-zoom: ;
    --tw-scroll-snap-strictness: proximity;
    --tw-ordinal: ;
    --tw-slashed-zero: ;
    --tw-numeric-figure: ;
    --tw-numeric-spacing: ;
    --tw-numeric-fraction: ;
    --tw-ring-inset: ;
    --tw-ring-offset-width: 0px;
    --tw-ring-offset-color: #fff;
    --tw-ring-color: rgb(59 130 246 / 0.5);
    --tw-ring-offset-shadow: 0 0 #0000;
    --tw-ring-shadow: 0 0 #0000;
    --tw-shadow: 0 0 #0000;
    --tw-shadow-colored: 0 0 #0000;
    --tw-blur: ;
    --tw-brightness: ;
    --tw-contrast: ;
    --tw-grayscale: ;
    --tw-hue-rotate: ;
    --tw-invert: ;
    --tw-saturate: ;
    --tw-sepia: ;
    --tw-drop-shadow: ;
    --tw-backdrop-blur: ;
    --tw-backdrop-brightness: ;
    --tw-backdrop-contrast: ;
    --tw-backdrop-grayscale: ;
    --tw-backdrop-hue-rotate: ;
    --tw-backdrop-invert: ;
    --tw-backdrop-opacity: ;
    --tw-backdrop-saturate: ;
    --tw-backdrop-sepia: ;
  }
  .container {
    width: 100%;
  }
  @media (min-width: 640px) {
    .container {
      max-width: 640px;
    }
  }
  @media (min-width: 768px) {
    .container {
      max-width: 768px;
    }
  }
  @media (min-width: 1024px) {
    .container {
      max-width: 1024px;
    }
  }
  @media (min-width: 1280px) {
    .container {
      max-width: 1280px;
    }
  }
  @media (min-width: 1536px) {
    .container {
      max-width: 1536px;
    }
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }
  .visible {
    visibility: visible;
  }
  .absolute {
    position: absolute;
  }
  .relative {
    position: relative;
  }
  .bottom-0 {
    bottom: 0px;
  }
  .left-0 {
    left: 0px;
  }
  .right-0 {
    right: 0px;
  }
  .right-2 {
    right: 0.5rem;
  }
  .right-2\.5 {
    right: 0.625rem;
  }
  .top-0 {
    top: 0px;
  }
  .top-2 {
    top: 0.5rem;
  }
  .top-2\.5 {
    top: 0.625rem;
  }
  .z-10 {
    z-index: 10;
  }
  .z-50 {
    z-index: 50;
  }
  .my-2 {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
  }
  .mb-2 {
    margin-bottom: 0.5rem;
  }
  .mb-4 {
    margin-bottom: 1rem;
  }
  .mr-2 {
    margin-right: 0.5rem;
  }
  .ml-1 {
    margin-left: 0.25rem;
  }
  .ml-2 {
    margin-left: 0.5rem;
  }
  .mr-5 {
    margin-right: 1.25rem;
  }
  .mt-2 {
    margin-top: 0.5rem;
  }
  .block {
    display: block;
  }
  .flex {
    display: flex;
  }
  .inline-flex {
    display: inline-flex;
  }
  .contents {
    display: contents;
  }
  .h-2 {
    height: 0.5rem;
  }
  .h-3 {
    height: 0.75rem;
  }
  .h-4 {
    height: 1rem;
  }
  .h-5 {
    height: 1.25rem;
  }
  .h-6 {
    height: 1.5rem;
  }
  .h-8 {
    height: 2rem;
  }
  .h-96 {
    height: 24rem;
  }
  .h-full {
    height: 100%;
  }
  .h-screen {
    height: 100vh;
  }
  .w-1\/2 {
    width: 50%;
  }
  .w-1\/3 {
    width: 33.333333%;
  }
  .w-11 {
    width: 2.75rem;
  }
  .w-2\/3 {
    width: 66.666667%;
  }
  .w-3 {
    width: 0.75rem;
  }
  .w-4 {
    width: 1rem;
  }
  .w-5 {
    width: 1.25rem;
  }
  .w-8 {
    width: 2rem;
  }
  .w-80 {
    width: 20rem;
  }
  .w-full {
    width: 100%;
  }
  .max-w-4xl {
    max-width: 56rem;
  }
  .rotate-90 {
    --tw-rotate: 90deg;
    transform: translate(var(--tw-translate-x), var(--tw-translate-y))
      rotate(var(--tw-rotate)) skew(var(--tw-skew-x)) skewY(var(--tw-skew-y))
      scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y));
  }
  .transform {
    transform: translate(var(--tw-translate-x), var(--tw-translate-y))
      rotate(var(--tw-rotate)) skew(var(--tw-skew-x)) skewY(var(--tw-skew-y))
      scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y));
  }
  .cursor-pointer {
    cursor: pointer;
  }
  .appearance-none {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
  }
  .flex-col {
    flex-direction: column;
  }
  .fill-current {
    fill: currentColor;
  }
  .flex-wrap {
    flex-wrap: wrap;
  }
  .items-center {
    align-items: center;
  }
  .justify-center {
    justify-content: center;
  }
  .gap-px {
    gap: 1px;
  }
  .space-x-1 > :not([hidden]) ~ :not([hidden]) {
    --tw-space-x-reverse: 0;
    margin-right: calc(0.25rem * var(--tw-space-x-reverse));
    margin-left: calc(0.25rem * calc(1 - var(--tw-space-x-reverse)));
  }
  .space-x-2 > :not([hidden]) ~ :not([hidden]) {
    --tw-space-x-reverse: 0;
    margin-right: calc(0.5rem * var(--tw-space-x-reverse));
    margin-left: calc(0.5rem * calc(1 - var(--tw-space-x-reverse)));
  }
  .divide-x > :not([hidden]) ~ :not([hidden]) {
    --tw-divide-x-reverse: 0;
    border-right-width: calc(1px * var(--tw-divide-x-reverse));
    border-left-width: calc(1px * calc(1 - var(--tw-divide-x-reverse)));
  }
  .divide-y > :not([hidden]) ~ :not([hidden]) {
    --tw-divide-y-reverse: 0;
    border-top-width: calc(1px * calc(1 - var(--tw-divide-y-reverse)));
    border-bottom-width: calc(1px * var(--tw-divide-y-reverse));
  }
  .overflow-hidden {
    overflow: hidden;
  }
  .overflow-y-auto {
    overflow-y: auto;
  }
  .rounded {
    border-radius: 0.25rem;
  }
  .rounded-full {
    border-radius: 9999px;
  }
  .rounded-lg {
    border-radius: 0.5rem;
  }
  .rounded-br {
    border-bottom-right-radius: 0.25rem;
  }
  .border {
    border-width: 1px;
  }
  .border-4 {
    border-width: 4px;
  }
  .border-b {
    border-bottom-width: 1px;
  }
  .border-l {
    border-left-width: 1px;
  }
  .border-r {
    border-right-width: 1px;
  }
  .border-t {
    border-top-width: 1px;
  }
  .border-solid {
    border-style: solid;
  }
  .border-dashed {
    border-style: dashed;
  }
  .border-gray-200 {
    --tw-border-opacity: 1;
    border-color: rgb(229 231 235 / var(--tw-border-opacity));
  }
  .border-gray-300 {
    --tw-border-opacity: 1;
    border-color: rgb(209 213 219 / var(--tw-border-opacity));
  }
  .border-gray-400 {
    --tw-border-opacity: 1;
    border-color: rgb(156 163 175 / var(--tw-border-opacity));
  }
  .border-white {
    --tw-border-opacity: 1;
    border-color: rgb(255 255 255 / var(--tw-border-opacity));
  }
  .bg-black {
    --tw-bg-opacity: 1;
    background-color: rgb(0 0 0 / var(--tw-bg-opacity));
  }
  .bg-blue-600 {
    --tw-bg-opacity: 1;
    background-color: rgb(37 99 235 / var(--tw-bg-opacity));
  }
  .bg-cyan-600 {
    --tw-bg-opacity: 1;
    background-color: rgb(8 145 178 / var(--tw-bg-opacity));
  }
  .bg-gray-100 {
    --tw-bg-opacity: 1;
    background-color: rgb(243 244 246 / var(--tw-bg-opacity));
  }
  .bg-gray-200 {
    --tw-bg-opacity: 1;
    background-color: rgb(229 231 235 / var(--tw-bg-opacity));
  }
  .bg-gray-50 {
    --tw-bg-opacity: 1;
    background-color: rgb(249 250 251 / var(--tw-bg-opacity));
  }
  .bg-gray-600 {
    --tw-bg-opacity: 1;
    background-color: rgb(75 85 99 / var(--tw-bg-opacity));
  }
  .bg-green-600 {
    --tw-bg-opacity: 1;
    background-color: rgb(22 163 74 / var(--tw-bg-opacity));
  }
  .bg-orange-600 {
    --tw-bg-opacity: 1;
    background-color: rgb(234 88 12 / var(--tw-bg-opacity));
  }
  .bg-red-600 {
    --tw-bg-opacity: 1;
    background-color: rgb(220 38 38 / var(--tw-bg-opacity));
  }
  .bg-transparent {
    background-color: transparent;
  }
  .bg-white {
    --tw-bg-opacity: 1;
    background-color: rgb(255 255 255 / var(--tw-bg-opacity));
  }
  .p-1 {
    padding: 0.25rem;
  }
  .p-1\.5 {
    padding: 0.375rem;
  }
  .p-2 {
    padding: 0.5rem;
  }
  .p-2\.5 {
    padding: 0.625rem;
  }
  .p-4 {
    padding: 1rem;
  }
  .p-5 {
    padding: 1.25rem;
  }
  .px-2 {
    padding-left: 0.5rem;
    padding-right: 0.5rem;
  }
  .px-4 {
    padding-left: 1rem;
    padding-right: 1rem;
  }
  .py-1 {
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
  }
  .py-2 {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
  }
  .py-4 {
    padding-top: 1rem;
    padding-bottom: 1rem;
  }
  .text-center {
    text-align: center;
  }
  .text-base {
    font-size: 1rem;
    line-height: 1.5rem;
  }
  .text-sm {
    font-size: 0.875rem;
    line-height: 1.25rem;
  }
  .text-xs {
    font-size: 0.75rem;
    line-height: 1rem;
  }
  .font-medium {
    font-weight: 500;
  }
  .font-semibold {
    font-weight: 600;
  }
  .text-gray-400 {
    --tw-text-opacity: 1;
    color: rgb(156 163 175 / var(--tw-text-opacity));
  }
  .text-gray-500 {
    --tw-text-opacity: 1;
    color: rgb(107 114 128 / var(--tw-text-opacity));
  }
  .text-gray-600 {
    --tw-text-opacity: 1;
    color: rgb(75 85 99 / var(--tw-text-opacity));
  }
  .text-gray-700 {
    --tw-text-opacity: 1;
    color: rgb(55 65 81 / var(--tw-text-opacity));
  }
  .text-gray-900 {
    --tw-text-opacity: 1;
    color: rgb(17 24 39 / var(--tw-text-opacity));
  }
  .text-orange-600 {
    --tw-text-opacity: 1;
    color: rgb(234 88 12 / var(--tw-text-opacity));
  }
  .invert {
    --tw-invert: invert(100%);
    filter: var(--tw-blur) var(--tw-brightness) var(--tw-contrast)
      var(--tw-grayscale) var(--tw-hue-rotate) var(--tw-invert)
      var(--tw-saturate) var(--tw-sepia) var(--tw-drop-shadow);
  }
  .transition-transform {
    transition-property: transform;
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
    transition-duration: 0.15s;
  }
  .scroll-hide {
    -ms-overflow-style: none;
    scrollbar-width: none;
  }
  .sr-only {
    clip: rect(0, 0, 0, 0);
    position: absolute;
    margin: -1px;
    border-width: 0;
    padding: 0;
    width: 1px;
    height: 1px;
    overflow: hidden;
    white-space: nowrap;
  }
  .scroll-hide::-webkit-scrollbar {
    display: none;
  }
  .gradio-container {
    -webkit-text-size-adjust: 100%;
    line-height: 1.5;
    font-family:
      Source Sans Pro,
      ui-sans-serif,
      system-ui,
      -apple-system,
      BlinkMacSystemFont,
      Segoe UI,
      Roboto,
      Helvetica Neue,
      Arial,
      Noto Sans,
      sans-serif,
      "Apple Color Emoji",
      "Segoe UI Emoji",
      Segoe UI Symbol,
      "Noto Color Emoji";
    -moz-tab-size: 4;
    -o-tab-size: 4;
    tab-size: 4;
  }
  .cropper-container {
    position: relative;
    touch-action: none;
    font-size: 0;
    line-height: 0;
    direction: ltr;
    -webkit-user-select: none;
    -moz-user-select: none;
    user-select: none;
  }
  .cropper-container img {
    display: block;
    image-orientation: 0deg;
    width: 100%;
    min-width: 0 !important;
    max-width: none !important;
    height: 100%;
    min-height: 0 !important;
    max-height: none !important;
  }
  .cropper-wrap-box,
  .cropper-canvas,
  .cropper-drag-box,
  .cropper-crop-box,
  .cropper-modal {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
  }
  .cropper-wrap-box,
  .cropper-canvas {
    overflow: hidden;
  }
  .cropper-drag-box {
    opacity: 0;
    background-color: #fff;
  }
  .cropper-modal {
    opacity: 0.5;
    background-color: #000;
  }
  .cropper-view-box {
    display: block;
    outline: 1px solid #39f;
    outline-color: #3399ffbf;
    width: 100%;
    height: 100%;
    overflow: hidden;
  }
  .cropper-dashed {
    display: block;
    position: absolute;
    opacity: 0.5;
    border: 0 dashed #eee;
  }
  .cropper-dashed.dashed-h {
    top: calc(100% / 3);
    left: 0;
    border-top-width: 1px;
    border-bottom-width: 1px;
    width: 100%;
    height: calc(100% / 3);
  }
  .cropper-dashed.dashed-v {
    top: 0;
    left: calc(100% / 3);
    border-right-width: 1px;
    border-left-width: 1px;
    width: calc(100% / 3);
    height: 100%;
  }
  .cropper-center {
    display: block;
    position: absolute;
    top: 50%;
    left: 50%;
    opacity: 0.75;
    width: 0;
    height: 0;
  }
  .cropper-center:before,
  .cropper-center:after {
    display: block;
    position: absolute;
    background-color: #eee;
    content: " ";
  }
  .cropper-center:before {
    top: 0;
    left: -3px;
    width: 7px;
    height: 1px;
  }
  .cropper-center:after {
    top: -3px;
    left: 0;
    width: 1px;
    height: 7px;
  }
  .cropper-face,
  .cropper-line,
  .cropper-point {
    display: block;
    position: absolute;
    opacity: 0.1;
    width: 100%;
    height: 100%;
  }
  .cropper-face {
    top: 0;
    left: 0;
    background-color: #fff;
  }
  .cropper-line {
    background-color: #39f;
  }
  .cropper-line.line-e {
    top: 0;
    right: -3px;
    cursor: ew-resize;
    width: 5px;
  }
  .cropper-line.line-n {
    top: -3px;
    left: 0;
    cursor: ns-resize;
    height: 5px;
  }
  .cropper-line.line-w {
    top: 0;
    left: -3px;
    cursor: ew-resize;
    width: 5px;
  }
  .cropper-line.line-s {
    bottom: -3px;
    left: 0;
    cursor: ns-resize;
    height: 5px;
  }
  .cropper-point {
    opacity: 0.75;
    background-color: #39f;
    width: 5px;
    height: 5px;
  }
  .cropper-point.point-e {
    top: 50%;
    right: -3px;
    cursor: ew-resize;
    margin-top: -3px;
  }
  .cropper-point.point-n {
    top: -3px;
    left: 50%;
    cursor: ns-resize;
    margin-left: -3px;
  }
  .cropper-point.point-w {
    top: 50%;
    left: -3px;
    cursor: ew-resize;
    margin-top: -3px;
  }
  .cropper-point.point-s {
    bottom: -3px;
    left: 50%;
    cursor: s-resize;
    margin-left: -3px;
  }
  .cropper-point.point-ne {
    top: -3px;
    right: -3px;
    cursor: nesw-resize;
  }
  .cropper-point.point-nw {
    top: -3px;
    left: -3px;
    cursor: nwse-resize;
  }
  .cropper-point.point-sw {
    bottom: -3px;
    left: -3px;
    cursor: nesw-resize;
  }
  .cropper-point.point-se {
    right: -3px;
    bottom: -3px;
    opacity: 1;
    cursor: nwse-resize;
    width: 20px;
    height: 20px;
  }
  @media (min-width: 768px) {
    .cropper-point.point-se {
      width: 15px;
      height: 15px;
    }
  }
  @media (min-width: 992px) {
    .cropper-point.point-se {
      width: 10px;
      height: 10px;
    }
  }
  @media (min-width: 1200px) {
    .cropper-point.point-se {
      opacity: 0.75;
      width: 5px;
      height: 5px;
    }
  }
  .cropper-point.point-se:before {
    display: block;
    position: absolute;
    right: -50%;
    bottom: -50%;
    opacity: 0;
    background-color: #39f;
    width: 200%;
    height: 200%;
    content: " ";
  }
  .cropper-invisible {
    opacity: 0;
  }
  .cropper-bg {
    background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQAQMAAAAlPW0iAAAAA3NCSVQICAjb4U/gAAAABlBMVEXMzMz////TjRV2AAAACXBIWXMAAArrAAAK6wGCiw1aAAAAHHRFWHRTb2Z0d2FyZQBBZG9iZSBGaXJld29ya3MgQ1M26LyyjAAAABFJREFUCJlj+M/AgBVhF/0PAH6/D/HkDxOGAAAAAElFTkSuQmCC);
  }
  .cropper-hide {
    display: block;
    position: absolute;
    width: 0;
    height: 0;
  }
  .cropper-hidden {
    display: none !important;
  }
  .cropper-move {
    cursor: move;
  }
  .cropper-crop {
    cursor: crosshair;
  }
  .cropper-disabled .cropper-drag-box,
  .cropper-disabled .cropper-face,
  .cropper-disabled .cropper-line,
  .cropper-disabled .cropper-point {
    cursor: not-allowed;
  }
  .after\:absolute:after {
    content: var(--tw-content);
    position: absolute;
  }
  .after\:left-\[2px\]:after {
    content: var(--tw-content);
    left: 2px;
  }
  .after\:top-0:after {
    content: var(--tw-content);
    top: 0px;
  }
  .after\:top-0\.5:after {
    content: var(--tw-content);
    top: 0.125rem;
  }
  .after\:h-5:after {
    content: var(--tw-content);
    height: 1.25rem;
  }
  .after\:w-5:after {
    content: var(--tw-content);
    width: 1.25rem;
  }
  .after\:rounded-full:after {
    content: var(--tw-content);
    border-radius: 9999px;
  }
  .after\:border:after {
    content: var(--tw-content);
    border-width: 1px;
  }
  .after\:border-gray-300:after {
    content: var(--tw-content);
    --tw-border-opacity: 1;
    border-color: rgb(209 213 219 / var(--tw-border-opacity));
  }
  .after\:bg-white:after {
    content: var(--tw-content);
    --tw-bg-opacity: 1;
    background-color: rgb(255 255 255 / var(--tw-bg-opacity));
  }
  .after\:transition-all:after {
    content: var(--tw-content);
    transition-property: all;
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
    transition-duration: 0.15s;
  }
  .after\:content-\[\'\'\]:after {
    --tw-content: "";
    content: var(--tw-content);
  }
  .hover\:bg-gray-200:hover {
    --tw-bg-opacity: 1;
    background-color: rgb(229 231 235 / var(--tw-bg-opacity));
  }
  .hover\:text-gray-600:hover {
    --tw-text-opacity: 1;
    color: rgb(75 85 99 / var(--tw-text-opacity));
  }
  .hover\:text-gray-900:hover {
    --tw-text-opacity: 1;
    color: rgb(17 24 39 / var(--tw-text-opacity));
  }
  .hover\:text-orange-600:hover {
    --tw-text-opacity: 1;
    color: rgb(234 88 12 / var(--tw-text-opacity));
  }
  .focus\:border-blue-500:focus {
    --tw-border-opacity: 1;
    border-color: rgb(59 130 246 / var(--tw-border-opacity));
  }
  .focus\:ring-blue-500:focus {
    --tw-ring-opacity: 1;
    --tw-ring-color: rgb(59 130 246 / var(--tw-ring-opacity));
  }
  .peer:checked ~ .peer-checked\:bg-orange-400 {
    --tw-bg-opacity: 1;
    background-color: rgb(251 146 60 / var(--tw-bg-opacity));
  }
  .peer:checked ~ .peer-checked\:after\:translate-x-full:after {
    content: var(--tw-content);
    --tw-translate-x: 100%;
    transform: translate(var(--tw-translate-x), var(--tw-translate-y))
      rotate(var(--tw-rotate)) skew(var(--tw-skew-x)) skewY(var(--tw-skew-y))
      scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y));
  }
  .peer:checked ~ .peer-checked\:after\:border-white:after {
    content: var(--tw-content);
    --tw-border-opacity: 1;
    border-color: rgb(255 255 255 / var(--tw-border-opacity));
  }
  .peer:focus ~ .peer-focus\:ring-4 {
    --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0
      var(--tw-ring-offset-width) var(--tw-ring-offset-color);
    --tw-ring-shadow: var(--tw-ring-inset) 0 0 0
      calc(4px + var(--tw-ring-offset-width)) var(--tw-ring-color);
    box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow),
      var(--tw-shadow, 0 0 #0000);
  }
  .peer:focus ~ .peer-focus\:ring-orange-300 {
    --tw-ring-opacity: 1;
    --tw-ring-color: rgb(253 186 116 / var(--tw-ring-opacity));
  }
  @media (prefers-color-scheme: dark) {
    .dark\:border-gray-600 {
      --tw-border-opacity: 1;
      border-color: rgb(75 85 99 / var(--tw-border-opacity));
    }
    .dark\:bg-gray-700 {
      --tw-bg-opacity: 1;
      background-color: rgb(55 65 81 / var(--tw-bg-opacity));
    }
    .dark\:bg-gray-800 {
      --tw-bg-opacity: 1;
      background-color: rgb(31 41 55 / var(--tw-bg-opacity));
    }
    .dark\:text-gray-400 {
      --tw-text-opacity: 1;
      color: rgb(156 163 175 / var(--tw-text-opacity));
    }
    .dark\:text-white {
      --tw-text-opacity: 1;
      color: rgb(255 255 255 / var(--tw-text-opacity));
    }
    .dark\:placeholder-gray-400::-moz-placeholder {
      --tw-placeholder-opacity: 1;
      color: rgb(156 163 175 / var(--tw-placeholder-opacity));
    }
    .dark\:placeholder-gray-400::placeholder {
      --tw-placeholder-opacity: 1;
      color: rgb(156 163 175 / var(--tw-placeholder-opacity));
    }
    .dark\:hover\:bg-gray-600:hover {
      --tw-bg-opacity: 1;
      background-color: rgb(75 85 99 / var(--tw-bg-opacity));
    }
    .dark\:hover\:text-white:hover {
      --tw-text-opacity: 1;
      color: rgb(255 255 255 / var(--tw-text-opacity));
    }
    .dark\:focus\:border-blue-500:focus {
      --tw-border-opacity: 1;
      border-color: rgb(59 130 246 / var(--tw-border-opacity));
    }
    .dark\:focus\:ring-blue-500:focus {
      --tw-ring-opacity: 1;
      --tw-ring-color: rgb(59 130 246 / var(--tw-ring-opacity));
    }
    .peer:focus ~ .dark\:peer-focus\:ring-orange-800 {
      --tw-ring-opacity: 1;
      --tw-ring-color: rgb(154 52 18 / var(--tw-ring-opacity));
    }
  }
  .minh {
    height: 600px;
  }

  .-top-1 {
    top: -0.25rem;
  }

  .left-6 {
    left: 1.5rem;
  }

  .h-4 {
    height: 1rem;
  }

  .w-4 {
    width: 1rem;
  }

  .w-max {
    width: max-content;
  }

  .cursor-pointer {
    cursor: pointer;
  }

  .rounded {
    border-radius: 0.25rem;
  }

  .bg-gray-900 {
    --tw-bg-opacity: 1;
    background-color: rgb(17 24 39 / var(--tw-bg-opacity));
  }

  .fill-current {
    fill: currentColor;
  }

  .p-10 {
    padding: 2.5rem;
  }

  .px-2 {
    padding-left: 0.5rem;
    padding-right: 0.5rem;
  }

  .py-1 {
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
  }

  .text-sm {
    font-size: 0.875rem;
    line-height: 1.25rem;
  }

  .font-medium {
    font-weight: 500;
  }

  .text-gray-50 {
    --tw-text-opacity: 1;
    color: rgb(249 250 251 / var(--tw-text-opacity));
  }

  .text-gray-500 {
    --tw-text-opacity: 1;
    color: rgb(107 114 128 / var(--tw-text-opacity));
  }

  .opacity-0 {
    opacity: 0;
  }

  .opacity-100 {
    opacity: 1;
  }

  .shadow {
    --tw-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
    --tw-shadow-colored: 0 1px 3px 0 var(--tw-shadow-color),
      0 1px 2px -1px var(--tw-shadow-color);
    box-shadow: var(--tw-ring-offset-shadow, 0 0 #0000),
      var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow);
  }

  .transition-opacity {
    transition-property: opacity;
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
    transition-duration: 150ms;
  }

  .hover\:text-orange-600:hover {
    --tw-text-opacity: 1;
    color: rgb(234 88 12 / var(--tw-text-opacity));
  }
</style>
