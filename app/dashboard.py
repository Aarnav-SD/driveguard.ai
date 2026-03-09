import streamlit as st
import random

st.title("DriveGuard AI")

score = random.randint(0,100)

st.metric("Driver Cognitive Alert Score",score)

if score > 70:
    st.success("Driver Alert")

elif score > 40:
    st.warning("Mild Cognitive Drift")

else:
    st.error("Drowsiness Detected")