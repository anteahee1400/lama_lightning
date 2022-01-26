import shortuuid
import time
from PIL import Image
import os
import time
import json
import io

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

def generate_id(length=8):
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(length)


def main():
    st.title("Inpainting Tool (LaMa)")
    device = st.sidebar.selectbox("device", ['0', '1', 'cpu'])
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 30, 17)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    # width = st.sidebar.number_input("width", value=600)
    # height = st.sidebar.number_input("height", value=400)
    run_id = generate_id()
    input_image = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
    canvas_result = None
    if input_image is not None:
        fname = input_image.name
        # input_image = Image.open(input_image).resize((width, height), Image.LANCZOS)
        input_image = Image.open(input_image)
        width = input_image.size[0]
        height = input_image.size[1]
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=input_image,
            initial_drawing=None,
            height=height,
            width=width,
            drawing_mode="freedraw",
            display_toolbar=True,
            update_streamlit=True,
        )
        
    l, m, r = st.columns(3)
    finish = l.button("Finish & Run")
    if finish:
        if canvas_result is not None:
            os.makedirs(f"{run_id}/data_for_prediction", exist_ok=True)
            os.makedirs(f"{run_id}/output", exist_ok=True)
            input_image.save(f"{run_id}/data_for_prediction/{fname}")
            mask = canvas_result.image_data[:,:,3]
            st.image(mask, caption="Mask result")
            fname_mask = f"{fname.split('.')[0]}_mask.png"
            plt.imsave(f"{run_id}/data_for_prediction/{fname_mask}", mask, cmap='gray')
            ext = os.path.splitext(fname)[-1]
            # print(ext)
            os.system(f"PYTHONPATH=. TORCH_HOME=$(pwd) python3 lama.py --indir=$(pwd)/{run_id}/data_for_prediction --outdir=$(pwd)/{run_id}/output --img_suffix={ext} --device={device}")
            result_img = None
            try:
                result_img = Image.open(f"{run_id}/output/{fname_mask}")
            except:
                pass
            
            if result_img is not None:
                st.text("Result Image")
                st.image(result_img)
                
                # os.system(f"rm -rf $(pwd)/{run_id}")
            # rerun = r.button("rerun")
            # if rerun:
            #     input_image = result_img
            
        else:
            st.text("please upload and mask your image")
            finish = False

if __name__ == "__main__":
    main()
