# web.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib, pickle
import shap
import matplotlib
import matplotlib.pyplot as plt

# 兼容 numpy 旧别名
if not hasattr(np, 'bool'):
    np.bool = bool

# ============== 字体/中文显示 ==================
def setup_chinese_font():
    """设置中文字体（优先系统字体，其次 ./fonts 目录）"""
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = [
            'WenQuanYi Zen Hei','WenQuanYi Micro Hei','SimHei','Microsoft YaHei',
            'PingFang SC','Hiragino Sans GB','Noto Sans CJK SC','Source Han Sans SC'
        ]
        available = [f.name for f in fm.fontManager.ttflist]
        for f in chinese_fonts:
            if f in available:
                matplotlib.rcParams['font.sans-serif'] = [f, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                return f

        # 尝试加载 ./fonts 下自带字体
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        candidates = [
            'NotoSansSC-Regular.otf','NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf','SimHei.ttf','MicrosoftYaHei.ttf'
        ]
        if os.path.isdir(fonts_dir):
            import matplotlib.font_manager as fm
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    fm.fontManager.addfont(fpath)
                    fam = fm.FontProperties(fname=fpath).get_name()
                    matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                    matplotlib.rcParams['font.family'] = 'sans-serif'
                    return fam
    except Exception:
        pass


    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = matplotlib.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============== Page configuration ==================
st.set_page_config(
    page_title="Gastrointestinal Diseases Patients Aged 60 - 85 Years' Depression Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ============== Feature names & display labels (ANN model) ==================
# These feature names must be consistent with the training data columns used in main.py
feature_names_display = [
    'Gender',
    'Pain',
    'Retire',
    'Falldown',
    'Disability',
    'Self_perceived_health',
    'Life_satisfaction',
    'Eyesight',
    'ADL_score',
    'Sleep_time'
]

feature_names_en = [
    'Gender',
    'Pain',
    'Retirement status',
    'History of falls',
    'Disability',
    'Self-perceived health',
    'Life satisfaction',
    'Eyesight',
    'ADL score (0-6)',
    'Sleep time (hours)'
]

feature_dict = dict(zip(feature_names_display, feature_names_en))

variable_descriptions = {
    'Gender': 'Biological sex (1: Male; 2: Female)',
    'Pain': 'Presence of pain (1: Yes; 0: No)',
    'Retire': 'Retirement status (1: Yes; 0: No)',
    'Falldown': 'Any previous history of falls (1: Yes; 0: No)',
    'Disability': 'Disability status (1: Yes; 0: No)',
    'Self_perceived_health': 'Self-perceived health (1: Poor; 2: Fair; 3: Good)',
    'Life_satisfaction': 'Life satisfaction (1: Poor; 2: Fair; 3: Good)',
    'Eyesight': 'Eyesight (1: Poor; 2: Fair; 3: Good)',
    'ADL_score': 'Activities of Daily Living score (0–6, higher = more impairment)',
    'Sleep_time': 'Average sleep duration per night (hours, no upper limit)'
}

# ============== 工具函数 ==================
def _clean_number(x):
    """把 '[3.3101046E-1]'、'3,210'、' 12. ' 等转成 float；失败返回 NaN"""
    if isinstance(x, str):
        s = x.strip().strip('[](){}').replace(',', '')
        try:
            return float(s)
        except Exception:
            return np.nan
    return x

@st.cache_resource
def load_model(model_path: str = './ann_model.pkl'):
    """Load the trained ANN model and infer its feature names if available."""
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        model_feature_names = None
        try:
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = list(model.feature_names_in_)
        except Exception:
            pass

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def predict_proba_safe(model, X_df):
    try:
        return model.predict_proba(X_df)
    except AttributeError:
        for k, v in {"use_label_encoder": False, "gpu_id": 0, "n_gpus": 1, "predictor": None}.items():
            if not hasattr(model, k):
                setattr(model, k, v)
        return model.predict_proba(X_df)
    except Exception:
        import xgboost as xgb
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is None:
            raise
        dm = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
        pred = booster.predict(dm, output_margin=False)
        if isinstance(pred, np.ndarray):
            if pred.ndim == 1:  
                proba_pos = pred.astype(float)
                return np.vstack([1 - proba_pos, proba_pos]).T
            elif pred.ndim == 2:
                return pred.astype(float)
        raise RuntimeError("Booster fallback failed: unknown output shape")

# ============== 主逻辑 ==================
def main():
    # Sidebar
    st.sidebar.title("Gastrointestinal Diseases Patients Aged 60 - 85 Years' Depression Risk Predictor")
    st.sidebar.image(
        "https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg",
        width=200
    )
    st.sidebar.markdown("""
    ### About
    This calculator uses an Artificial Neural Network (ANN) model to predict
    the risk of depression in patients aged 60–85 years with gastrointestinal diseases.

    **Outputs:**
    - Predicted probability of depression vs. no depression
    - SHAP-based model explanation

    """)
    with st.sidebar.expander("Variable description"):
        for f in feature_names_display:
            st.markdown(f"**{feature_dict.get(f, f)}**: {variable_descriptions.get(f, '')}")

    st.title("Gastrointestinal Diseases Patients Aged 60 - 85 Years' Depression Risk Predictor")


    # Load ANN model
    try:
        model, model_feature_names = load_model('./ann_model.pkl')
        st.sidebar.success("ANN model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return

    # 输入区域
    st.header("Patient characteristics")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox(
            "Gender",
            options=[1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female",
            index=0
        )
        pain = st.selectbox(
            "Pain",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            index=0
        )
        retire = st.selectbox(
            "Retirement status",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            index=0
        )

    with c2:
        falldown = st.selectbox(
            "History of falls",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            index=0
        )
        disability = st.selectbox(
            "Disability",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            index=0
        )
        self_health = st.selectbox(
            "Self-perceived health",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Poor", 2: "Fair", 3: "Good"}.get(x, str(x)),
            index=1
        )

    with c3:
        life_satis = st.selectbox(
            "Life satisfaction",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Poor", 2: "Fair", 3: "Good"}.get(x, str(x)),
            index=1
        )
        eyesight = st.selectbox(
            "Eyesight",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Poor", 2: "Fair", 3: "Good"}.get(x, str(x)),
            index=2
        )
        adl_score = st.number_input(
            "ADL score (0-6)",
            min_value=0.0,
            max_value=6.0,
            value=0.0,
            step=1.0
        )

    c4, _ = st.columns([1, 1])
    with c4:
        sleep_time = st.number_input(
            "Sleep time per night (hours)",
            min_value=0.0,
            value=7.0,
            step=0.5
        )

    if st.button("Run prediction", type="primary"):
        # Assemble input according to training features
        user_inputs = {
            'Gender': gender,
            'Pain': pain,
            'Retire': retire,
            'Falldown': falldown,
            'Disability': disability,
            'Self_perceived_health': self_health,
            'Life_satisfaction': life_satis,
            'Eyesight': eyesight,
            'ADL_score': adl_score,
            'Sleep_time': sleep_time,
        }

        # Construct input DataFrame in the exact order expected by the ANN model
        if model_feature_names:
            try:
                row = [user_inputs[c] for c in model_feature_names]
            except KeyError as e:
                st.error(f"Feature name mismatch between model and UI: {e}")
                with st.expander("Debug: model vs UI features"):
                    st.write("Model feature names:", model_feature_names)
                    st.write("UI input keys:", list(user_inputs.keys()))
                return
            input_df = pd.DataFrame([row], columns=model_feature_names)
        else:
            input_df = pd.DataFrame([[user_inputs[c] for c in feature_names_display]], columns=feature_names_display)

        # Clean & convert to numeric
        input_df = input_df.applymap(_clean_number)
        for c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
        if input_df.isnull().any().any():
            st.error("There are missing or unparsable input values. Please check that all numeric fields are valid numbers.")
            with st.expander("Debug: current input DataFrame"):
                st.write(input_df)
            return

        # ======== Prediction ========
        try:
            proba = predict_proba_safe(model, input_df)[0]
            if len(proba) == 2:
                no_depress_prob = float(proba[0]); depress_prob = float(proba[1])
            else:
                raise ValueError("返回的概率维度异常")

            # 展示结果
            st.header("Depression risk prediction result")
            a, b = st.columns(2)
            with a:
                st.subheader("Probability of no depression")
                st.progress(no_depress_prob)
                st.write(f"{no_depress_prob:.2%}")
            with b:
                st.subheader("Probability of depression")
                st.progress(depress_prob)
                st.write(f"{depress_prob:.2%}")

            # ======= SHAP explanation (ANN, using KernelExplainer + waterfall_plot) =======
            st.write("---")
            st.subheader("Model explanation (SHAP waterfall plot)")

            try:
                # Use KernelExplainer on the positive-class probability (depression)
                f = lambda X: model.predict_proba(X)[:, 1]
                background = input_df.copy()
                explainer = shap.KernelExplainer(f, background)
                shap_values = explainer.shap_values(input_df, nsamples="auto")

                # KernelExplainer for binary classification returns a 1D array for class 1
                shap_value = np.array(shap_values[0])
                feature_names_current = list(input_df.columns)

                # Limit to top features to keep the figure compact
                order = np.argsort(np.abs(shap_value))[::-1]
                max_display = min(10, len(order))
                shap_value_ordered = shap_value[order]
                data_ordered = input_df.iloc[0, order]
                feature_names_ordered = [feature_names_current[i] for i in order]

                # Compute expected value from explainer
                expected_value = explainer.expected_value

                # Create a small figure and draw SHAP waterfall plot
                plt.figure(figsize=(6, 4))
                try:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value_ordered,
                            base_values=expected_value,
                            data=data_ordered.values,
                            feature_names=feature_names_ordered,
                        ),
                        max_display=max_display,
                        show=False,
                    )
                except Exception:
                    # fallback without feature name reordering if anything goes wrong
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=input_df.iloc[0].values,
                            feature_names=feature_names_current,
                        ),
                        max_display=max_display,
                        show=False,
                    )

                fig = plt.gcf()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.warning(f"Failed to generate SHAP explanation: {e}")
                import traceback; st.text(traceback.format_exc())

        except Exception as e:
            st.error(f"Prediction or result display failed: {e}")
            import traceback; st.error(traceback.format_exc())

    st.write("---")
    st.caption("Depression Risk Calculator (ANN model)")

if __name__ == "__main__":
    main()
