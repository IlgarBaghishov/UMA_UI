
import gradio as gr
from gradio_molecule3d import Molecule3D


example = Molecule3D().example_value()


reps =    [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "cartoon",
      "color": "hydrophobicity",
      # "residue_range": "",
      "around": 0,
      "byres": False,
      # "visible": False,
      # "opacity": 0.5
    }
  ]



def predict(x):
    print("predict function", x)
    print(x.name)
    return x

# def update_color(mol, color):
#     reps[0]['color'] = color
#     print(reps)
#     return Molecule3D(mol, reps=reps)

with gr.Blocks() as demo:
    gr.Markdown("# Molecule3D")
    # color_choices = ['redCarbon', 'greenCarbon', 'orangeCarbon', 'blackCarbon', 'blueCarbon', 'grayCarbon', 'cyanCarbon']

    inp = Molecule3D(label="Molecule3D", reps=reps)
    # cdr_color = gr.Dropdown(choices=color_choices, label="CDR color", value='redCarbon')
    out = Molecule3D(label="Output", reps=reps)
    # cdr_color.change(update_color, inputs=[inp,cdr_color], outputs=out)
    btn = gr.Button("Predict")
    gr.Markdown(""" 
    You can configure the default rendering of the molecule by adding a list of representations
    <pre>
        reps =    [
        {
          "model": 0,
          "style": "cartoon",
          "color": "whiteCarbon",
          "residue_range": "",
          "around": 0,
          "opacity":1
          
        },
        {
          "model": 0,
          "chain": "A",
          "resname": "HIS",
          "style": "stick",
          "color": "red"
        }
      ]
    </pre>
    """)
    btn.click(predict, inputs=inp, outputs=out)


if __name__ == "__main__":
    demo.launch()
