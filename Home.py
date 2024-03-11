
import streamlit as st
import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cloudpickle import cloudpickle

# import mpld3
from sdk_model.create_lm_from_2points import RebeccaCustomModel, create_data_from_2_points, get_fit_metrics


st.set_page_config(page_title="SaaS - ProfessIUR.ai", page_icon=":house", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title('Creazione modelli con SDK di Rebecca per Ottimizzatore NIA')



