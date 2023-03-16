import streamlit as st
import numpy as np
import pandas as pd
from streamlit_player import st_player
import joblib as jl

from PIL import Image
image = Image.open('media_files/GTA.png')
st.image(image, caption='GTA Mortgage approval form', use_column_width=True)


# Code below is needed for the next and previous buttons
from streamlit.components.v1 import html
def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)
if st.button("credit"):
    nav_page("credit")
if st.button("form"):
    nav_page("form")
