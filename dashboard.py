import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from scipy.integrate import solve_ivp
from numba import njit

# Set page configuration for a premium visual aesthetic
st.set_page_config(
    page_title="Stiff Adaptive Predator-Prey ODE Dashboard",
    page_icon="🦎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and styling tweaks
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #1e293b !important;
    }
    p, span, div, label, li {
        color: #1e293b;
    }
    .stSidebar {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }
    h1, h2, h3 {
        color: #0f172a !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        color: #0f172a;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. MATHEMATICAL MODEL & SOLVERS (JIT-COMPILED)
# ----------------------------------------------------

@njit(cache=True)
def system_func(t, v, epsilon, extinction_threshold=0.001):
    """
    RHS function of the 3D adaptive predator-prey system.
    v = [x, y, alpha2]
    """
    x, y, a2 = v
    # Regularization to prevent division by zero or negative adaptivity
    a2_reg = a2 if abs(a2) > 1e-12 else 1e-12
    
    dxdt = x * (1.0 - 0.5 * x - (2.0/7.0) * (a2_reg**-2) * y)
    dydt = y * (2.0 * a2 - 0.5 * y - 3.5 * (a2_reg**2) * x)
    da2dt = epsilon * (2.0 - 7.0 * a2 * x)
    
    # Apply extinction thresholds
    if x < extinction_threshold:
        dxdt = 0.0
    if y < extinction_threshold:
        dydt = 0.0
    if a2 < extinction_threshold:
        da2dt = 0.0
        
    return np.array([dxdt, dydt, da2dt])

@njit(cache=True)
def system_jacobian(state, epsilon, extinction_threshold=0.001):
    """
    Analytical Jacobian matrix of the RHS.
    Used for the Implicit Euler Newton corrector and stiffness calculation.
    Includes extinction threshold clipping to maintain mathematical consistency with the RHS.
    """
    x, y, alpha2 = state
    # Regularization
    alpha2_reg = alpha2 if alpha2 > 1e-12 else 1e-12
    
    J = np.zeros((3, 3))
    
    # Check extinction thresholds and zero out the rows of Jacobian if extinct
    if x >= extinction_threshold:
        J[0, 0] = 1.0 - x - (2.0/7.0) * (alpha2_reg**-2) * y
        J[0, 1] = -(2.0/7.0) * (alpha2_reg**-2) * x
        J[0, 2] = (4.0/7.0) * (alpha2_reg**-3) * x * y
        
    if y >= extinction_threshold:
        J[1, 0] = -3.5 * (alpha2_reg**2) * y
        J[1, 1] = 2.0 * alpha2_reg - y - 3.5 * (alpha2_reg**2) * x
        J[1, 2] = 2.0 * y - 7.0 * alpha2_reg * x * y
        
    if alpha2 >= extinction_threshold:
        J[2, 0] = -7.0 * epsilon * alpha2_reg
        J[2, 1] = 0.0
        J[2, 2] = -7.0 * epsilon * x
    
    return J

@njit(cache=True)
def rk4_step_fast(t, y, h, epsilon, extinction_threshold):
    k1 = system_func(t, y, epsilon, extinction_threshold)
    k2 = system_func(t + h/2.0, y + h/2.0 * k1, epsilon, extinction_threshold)
    k3 = system_func(t + h/2.0, y + h/2.0 * k2, epsilon, extinction_threshold)
    k4 = system_func(t + h, y + h * k3, epsilon, extinction_threshold)
    return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

@njit(cache=True)
def euler_explicit(y0, t_span, h, epsilon, extinction_threshold):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 3))
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i+1] = y[i] + h * system_func(t[i], y[i], epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])):
            return t[:i+2], y[:i+2]
    return t, y

@njit(cache=True)
def rk4_solver(y0, t_span, h, epsilon, extinction_threshold):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 3))
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i+1] = rk4_step_fast(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])):
            return t[:i+2], y[:i+2]
    return t, y

@njit(cache=True)
def implicit_euler_newton(y0, t_span, h, epsilon, extinction_threshold, tol=1e-7, max_iter=100):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 3))
    y[0] = y0
    
    I = np.eye(3)
    
    for i in range(n_steps - 1):
        # Predictor (explicit Euler step)
        y_guess = y[i] + h * system_func(t[i], y[i], epsilon, extinction_threshold)
        
        # Corrector (Newton iterations)
        converged = False
        for _ in range(max_iter):
            F = y_guess - y[i] - h * system_func(t[i+1], y_guess, epsilon, extinction_threshold)
            J_f = system_jacobian(y_guess, epsilon, extinction_threshold)
            J_F = I - h * J_f
            
            if np.any(np.isnan(J_F)) or np.any(np.isinf(J_F)) or np.any(np.isnan(F)) or np.any(np.isinf(F)):
                break
                
            try:
                dy = np.linalg.solve(J_F, -F)
            except:
                break
                
            y_guess = y_guess + dy
            if np.linalg.norm(dy) < tol:
                converged = True
                break
        
        y[i+1] = y_guess
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])):
            return t[:i+2], y[:i+2]
            
    return t, y

# ----------------------------------------------------
# 2. SCIPY INTEGRATION WRAPPER
# ----------------------------------------------------

def scipy_solver_wrapper(y0, t_span, h, epsilon, extinction_threshold, method='BDF', tol=1e-12):
    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    t_eval = t_eval[t_eval <= t_span[1]]
    
    def rhs_scipy(t, y):
        return system_func(t, y, epsilon, extinction_threshold)
        
    sol = solve_ivp(
        rhs_scipy,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        rtol=tol,
        atol=tol
    )
    return sol.t, sol.y.T

# ----------------------------------------------------
# 3. STIFFNESS & EIGENVALUES CALCULATION (SYSTEM 1)
# ----------------------------------------------------

def compute_stiffness_and_eigenvalues(t, y, epsilon, extinction_threshold=0.001):
    """
    Computes real parts of eigenvalues of the Jacobian along a trajectory (System 1).
    Calculates stiffness ratio L(t) = max|Re(lambda_i)| / min|Re(lambda_i)|.
    """
    eigvals_real = []
    stiffness_ratios = []
    
    for i in range(len(t)):
        state = y[i]
        J = system_jacobian(state, epsilon, extinction_threshold)
        try:
            evs = np.linalg.eigvals(J)
            evs_real = np.real(evs)
            eigvals_real.append(evs_real)
            
            abs_re = np.abs(evs_real)
            min_val = np.min(abs_re)
            max_val = np.max(abs_re)
            if min_val < 1e-15:
                min_val = 1e-15
            ratio = max_val / min_val
            stiffness_ratios.append(ratio)
        except:
            eigvals_real.append(np.array([np.nan, np.nan, np.nan]))
            stiffness_ratios.append(np.nan)
            
    return np.array(eigvals_real), np.array(stiffness_ratios)

# ----------------------------------------------------
# 1B. SYSTEM 2 MATHEMATICAL MODEL & SOLVERS (4D DOUBLE-ADAPTIVE)
# ----------------------------------------------------

@njit(cache=True)
def system2_func(t, v, epsilon, extinction_threshold=0.001):
    x, y, a1, a2 = v
    a1_reg = a1 if abs(a1) > 1e-12 else 1e-12
    a2_reg = a2 if abs(a2) > 1e-12 else 1e-12
    
    dxdt = x * (2.0 * a1 - 0.5 * x - (a1_reg**2) * (a2_reg**-2) * y)
    dydt = y * (2.0 * a2 - 0.5 * y - (a1_reg**-2) * (a2_reg**2) * x)
    da1dt = epsilon * (2.0 - 2.0 * a1 * (a2_reg**-2) * y)
    da2dt = epsilon * (2.0 - 2.0 * (a1_reg**-2) * a2 * x)
    
    if x < extinction_threshold:
        dxdt = 0.0
    if y < extinction_threshold:
        dydt = 0.0
    if a1 < extinction_threshold:
        da1dt = 0.0
    if a2 < extinction_threshold:
        da2dt = 0.0
        
    return np.array([dxdt, dydt, da1dt, da2dt])

@njit(cache=True)
def system2_jacobian(state, epsilon, extinction_threshold=0.001):
    x, y, a1, a2 = state
    a1_reg = a1 if a1 > 1e-12 else 1e-12
    a2_reg = a2 if a2 > 1e-12 else 1e-12
    
    J = np.zeros((4, 4))
    
    if x >= extinction_threshold:
        J[0, 0] = 2.0 * a1_reg - x - (a1_reg**2) * (a2_reg**-2) * y
        J[0, 1] = -(a1_reg**2) * (a2_reg**-2) * x
        J[0, 2] = 2.0 * x - 2.0 * a1_reg * (a2_reg**-2) * x * y
        J[0, 3] = 2.0 * (a1_reg**2) * (a2_reg**-3) * x * y
        
    if y >= extinction_threshold:
        J[1, 0] = -(a1_reg**-2) * (a2_reg**2) * y
        J[1, 1] = 2.0 * a2_reg - y - (a1_reg**-2) * (a2_reg**2) * x
        J[1, 2] = 2.0 * (a1_reg**-3) * (a2_reg**2) * x * y
        J[1, 3] = 2.0 * y - 2.0 * (a1_reg**-2) * a2_reg * x * y
        
    if a1 >= extinction_threshold:
        J[2, 0] = 0.0
        J[2, 1] = -2.0 * epsilon * a1_reg * (a2_reg**-2)
        J[2, 2] = -2.0 * epsilon * (a2_reg**-2) * y
        J[2, 3] = 4.0 * epsilon * a1_reg * (a2_reg**-3) * y
        
    if a2 >= extinction_threshold:
        J[3, 0] = -2.0 * epsilon * (a1_reg**-2) * a2_reg
        J[3, 1] = 0.0
        J[3, 2] = 4.0 * epsilon * (a1_reg**-3) * a2_reg * x
        J[3, 3] = -2.0 * epsilon * (a1_reg**-2) * x
        
    return J

@njit(cache=True)
def rk4_step_fast2(t, y, h, epsilon, extinction_threshold):
    k1 = system2_func(t, y, epsilon, extinction_threshold)
    k2 = system2_func(t + h/2.0, y + h/2.0 * k1, epsilon, extinction_threshold)
    k3 = system2_func(t + h/2.0, y + h/2.0 * k2, epsilon, extinction_threshold)
    k4 = system2_func(t + h, y + h * k3, epsilon, extinction_threshold)
    return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

@njit(cache=True)
def euler_explicit2(y0, t_span, h, epsilon, extinction_threshold):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 4))
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i+1] = y[i] + h * system2_func(t[i], y[i], epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])):
            return t[:i+2], y[:i+2]
    return t, y

@njit(cache=True)
def rk4_solver2(y0, t_span, h, epsilon, extinction_threshold):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 4))
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i+1] = rk4_step_fast2(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])):
            return t[:i+2], y[:i+2]
    return t, y

@njit(cache=True)
def implicit_euler_newton2(y0, t_span, h, epsilon, extinction_threshold, tol=1e-7, max_iter=100):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 4))
    y[0] = y0
    
    I = np.eye(4)
    
    for i in range(n_steps - 1):
        y_guess = y[i] + h * system2_func(t[i], y[i], epsilon, extinction_threshold)
        
        converged = False
        for _ in range(max_iter):
            F = y_guess - y[i] - h * system2_func(t[i+1], y_guess, epsilon, extinction_threshold)
            J_f = system2_jacobian(y_guess, epsilon, extinction_threshold)
            J_F = I - h * J_f
            
            if np.any(np.isnan(J_F)) or np.any(np.isinf(J_F)) or np.any(np.isnan(F)) or np.any(np.isinf(F)):
                break
                
            try:
                dy = np.linalg.solve(J_F, -F)
            except:
                break
                
            y_guess = y_guess + dy
            if np.linalg.norm(dy) < tol:
                converged = True
                break
        
        y[i+1] = y_guess
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])):
            return t[:i+2], y[:i+2]
            
    return t, y

def scipy_solver_wrapper2(y0, t_span, h, epsilon, extinction_threshold, method='BDF', tol=1e-12):
    t_eval = np.arange(t_span[0], t_span[1] + h, h)
    t_eval = t_eval[t_eval <= t_span[1]]
    
    def rhs_scipy(t, y):
        return system2_func(t, y, epsilon, extinction_threshold)
        
    sol = solve_ivp(
        rhs_scipy,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        rtol=tol,
        atol=tol
    )
    return sol.t, sol.y.T

def compute_stiffness_and_eigenvalues2(t, y, epsilon, extinction_threshold=0.001):
    """
    Computes real parts of eigenvalues of the Jacobian along a trajectory (System 2).
    Calculates stiffness ratio L(t) = max|Re(lambda_i)| / min|Re(lambda_i)|.
    """
    eigvals_real = []
    stiffness_ratios = []
    
    for i in range(len(t)):
        state = y[i]
        J = system2_jacobian(state, epsilon, extinction_threshold)
        try:
            evs = np.linalg.eigvals(J)
            evs_real = np.real(evs)
            eigvals_real.append(evs_real)
            
            abs_re = np.abs(evs_real)
            min_val = np.min(abs_re)
            max_val = np.max(abs_re)
            if min_val < 1e-15:
                min_val = 1e-15
            ratio = max_val / min_val
            stiffness_ratios.append(ratio)
        except:
            eigvals_real.append(np.array([np.nan, np.nan, np.nan, np.nan]))
            stiffness_ratios.append(np.nan)
            
    return np.array(eigvals_real), np.array(stiffness_ratios)

# ----------------------------------------------------
# 4. STREAMLIT INTERFACE & ROUTING
# ----------------------------------------------------

# st.sidebar.title("⚙️ Конфигурация Модели")
st.sidebar.markdown("---")

st.sidebar.subheader("Выбор системы ODE")
model_selection = st.sidebar.radio(
    "Выберите математическую модель",
    ["Система 1 (3D Адаптивная)", "Система 2 (4D Адаптивная)"]
)

# Render title and math based on selection
if model_selection == "Система 1 (3D Адаптивная)":
    st.title("🦎 3D Адаптивная Жесткая Модель Хищник-Жертва")
    st.markdown(r"""
    На этой вкладке моделируется 3D жесткая система хищник-жертва с адаптивной эффективностью поиска хищника $\alpha_2$:
    $$
    \frac{dx}{dt} = x \left( 1 - 0.5x - \frac{2}{7} \alpha_2^{-2} y \right), \quad \\[1ex]
    \frac{dy}{dt} = y \left( 2\alpha_2 - 0.5y - 3.5\alpha_2^2 x \right), \quad \\[1ex]
    \frac{d\alpha_2}{dt} = \varepsilon (2 - 7\alpha_2 x)\\[1ex]
    $$
    Когда $\alpha_2$ близка к нулю, $\alpha_2^{-2}$ становится очень большой, что делает систему сильно жесткой.
    """)
else:
    st.title("🦕 4D Дважды-Адаптивная Жесткая Модель Хищник-Жертва")
    st.markdown(r"""
    На этой вкладке моделируется 4D жесткая система хищник-жертва с двумя адаптивными признаками $\alpha_1$ и $\alpha_2$:
    $$
    \begin{align*}
    \frac{dx}{dt} &= x\left(2\alpha_1 - 0.5x - \alpha_1^2 \alpha_2^{-2} y\right), \quad x(0) = x_0, \\[1ex]
    \frac{dy}{dt} &= y\left(2\alpha_2 - 0.5y - \alpha_1^{-2} \alpha_2^2 x\right), \quad y(0) = y_0, \\[1ex]
    \frac{d\alpha_1}{dt} &= \varepsilon\left(2 - 2\alpha_1 \alpha_2^{-2} y\right), \quad \alpha_1(0) = \alpha_{10}, \\[1ex]
    \frac{d\alpha_2}{dt} &= \varepsilon\left(2 - 2\alpha_1^{-2} \alpha_2 x\right), \quad \alpha_2(0) = \alpha_{20}; \qquad t \in [0; T_k].
    \end{align*}
    $$
    """)

# Sidebar controls
st.sidebar.header("🔧 Панель конфигурации")

# Helper function for synchronized slider + number input inside sidebar columns
def slider_and_input(label, min_val, max_val, default_val, step_val, fmt=None, key_pref=""):
    slider_key = f"{key_pref}_slider"
    input_key = f"{key_pref}_input"
    
    # Initialize values in session state if not present
    if slider_key not in st.session_state:
        st.session_state[slider_key] = float(default_val)
    if input_key not in st.session_state:
        st.session_state[input_key] = float(default_val)
        
    def on_slider_change():
        st.session_state[input_key] = st.session_state[slider_key]
        
    def on_input_change():
        val = st.session_state[input_key]
        val = max(float(min_val), min(float(max_val), val))
        st.session_state[input_key] = val
        st.session_state[slider_key] = val

    col1, col2 = st.sidebar.columns([3, 2])
    with col1:
        val_slider = st.slider(
            label, 
            min_value=float(min_val), 
            max_value=float(max_val), 
            step=float(step_val), 
            key=slider_key, 
            on_change=on_slider_change
        )
    with col2:
        val_input = st.number_input(
            "ввести:", 
            min_value=float(min_val), 
            max_value=float(max_val), 
            step=float(step_val), 
            format=fmt, 
            key=input_key, 
            on_change=on_input_change
        )
    return val_slider

# Render sidebar parameters dynamically based on model selection
if model_selection == "Система 1 (3D Адаптивная)":
    st.sidebar.subheader("Параметры системы")
    eps = slider_and_input("Epsilon (ε - Скорость адапт.)", 0.001, 0.1, 0.05, 0.0001, fmt="%.3f", key_pref="eps_s1")
    extinction_thresh = slider_and_input("Порог вымирания", 0.0, 0.1, 0.001, 0.0005, fmt="%.4f", key_pref="extinction_s1")
    
    st.sidebar.subheader("Начальные условия")
    x0 = slider_and_input("x0 (Жертва)", 0.0, 5.0, 0.4, 0.1, fmt="%.2f", key_pref="x0_s1")
    y0 = slider_and_input("y0 (Хищник)", 0.0, 20.0, 1.4, 0.5, fmt="%.2f", key_pref="y0_s1")
    alpha20 = slider_and_input("α20 (Адаптивность)", 0.00001, 1.0, 0.5, 0.0001, fmt="%.5f", key_pref="alpha20_s1")
    
    st.sidebar.subheader("Настройки симуляции")
    Tk = slider_and_input("Tk (Конечное время)", 1.0, 3000.0, 1500.0, 10.0, fmt="%.1f", key_pref="Tk_s1")
    exp_h = slider_and_input("log10(h) (Шаг по времени)", -5, 0, -1, 0.5, fmt="%.1f", key_pref="h_exp_s1")
    h = 10 ** exp_h
else:
    st.sidebar.subheader("Параметры системы")
    eps = slider_and_input("Epsilon (ε - Скорость адапт.)", 0.001, 0.1, 0.01, 0.0001, fmt="%.3f", key_pref="eps_s2")
    extinction_thresh = slider_and_input("Порог вымирания", 0.0, 0.1, 0.001, 0.0005, fmt="%.4f", key_pref="extinction_s2")
    
    st.sidebar.subheader("Начальные условия")
    x0 = slider_and_input("x0 (Жертва)", 0.0, 50.0, 10.0, 0.5, fmt="%.1f", key_pref="x0_s2")
    y0 = slider_and_input("y0 (Хищник)", 0.0, 50.0, 10.0, 0.5, fmt="%.1f", key_pref="y0_s2")
    alpha10 = slider_and_input("α10 (Адапт. хищника)", 0.00001, 1.0, 0.0001, 0.0001, fmt="%.5f", key_pref="alpha10_s2")
    alpha20 = slider_and_input("α20 (Адапт. жертвы)", 0.1, 20.0, 10.0, 0.5, fmt="%.2f", key_pref="alpha20_s2")
    
    st.sidebar.subheader("Настройки симуляции")
    Tk = slider_and_input("Tk (Конечное время)", 1.0, 3000.0, 2000.0, 10.0, fmt="%.1f", key_pref="Tk_s2")
    exp_h = slider_and_input("log10(h) (Шаг по времени)", -5, 0, -1, 0.5, fmt="%.1f", key_pref="h_exp_s2")
    h = 10 ** exp_h

solver_name = st.sidebar.selectbox(
    "Выберите метод решения (Солвер)",
    [
        "Implicit Euler (Newton, Custom)",
        "SciPy BDF (Stiff)",
        "SciPy Radau (Stiff)",
        "SciPy LSODA (Stiff/Non-Stiff)",
        "SciPy RK45 (Explicit)",
        "Runge-Kutta 4 (Explicit, Custom)",
        "Explicit Euler (Custom)",
        "Adams-Bashforth 4 (Explicit, Custom)",
        "Adams-Moulton 4 (Implicit, Custom)",
        "Adams-Bashforth-Moulton 4 (Predictor-Corrector, Custom)",
        "Gear 4 / BDF4 (Implicit, Custom)"
    ]
)

# Run Simulation (cached in Streamlit to avoid re-solving on UI updates)
# ----------------------------------------------------
# 6. CUSTOM SOLVERS FROM LAB 3
# ----------------------------------------------------
@njit(cache=True)
def adams_bashfort_4_solver(y0, t_span, h, epsilon, extinction_threshold):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 3))
    y[0] = y0
    if n_steps == 1: return t, y
    for i in range(min(3, n_steps - 1)):
        y[i+1] = rk4_step_fast(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]

    for i in range(3, n_steps - 1):
        f0 = system_func(t[i], y[i], epsilon, extinction_threshold)
        f1 = system_func(t[i-1], y[i-1], epsilon, extinction_threshold)
        f2 = system_func(t[i-2], y[i-2], epsilon, extinction_threshold)
        f3 = system_func(t[i-3], y[i-3], epsilon, extinction_threshold)
        y[i+1] = y[i] + h/24.0 * (55.0*f0 - 59.0*f1 + 37.0*f2 - 9.0*f3)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
    return t, y

@njit(cache=True)
def adams_molton_4_solver(y0, t_span, h, epsilon, extinction_threshold, tol=1e-7, max_iter=100):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 3))
    y[0] = y0
    if n_steps == 1: return t, y
    for i in range(min(3, n_steps - 1)):
        y[i+1] = rk4_step_fast(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
        
    if n_steps <= 4: return t, y
    
    I = np.eye(3)
    f0 = system_func(t[3], y[3], epsilon, extinction_threshold)
    f1 = system_func(t[2], y[2], epsilon, extinction_threshold)
    f2 = system_func(t[1], y[1], epsilon, extinction_threshold)
    
    for i in range(3, n_steps - 1):
        y_guess = y[i].copy()
        converged = False
        for _ in range(max_iter):
            fz = system_func(t[i+1], y_guess, epsilon, extinction_threshold)
            F = y_guess - y[i] - (h / 24.0) * (9.0 * fz + 19.0 * f0 - 5.0 * f1 + f2)
            J_f = system_jacobian(y_guess, epsilon, extinction_threshold)
            J_F = I - (h * 9.0 / 24.0) * J_f
            if np.any(np.isnan(J_F)) or np.any(np.isinf(J_F)) or np.any(np.isnan(F)) or np.any(np.isinf(F)): break
            try: dy = np.linalg.solve(J_F, -F)
            except: break
            y_guess = y_guess + dy
            if np.linalg.norm(dy) < tol:
                converged = True
                break
        y[i+1] = y_guess
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
        f2 = f1
        f1 = f0
        f0 = system_func(t[i+1], y[i+1], epsilon, extinction_threshold)
    return t, y

@njit(cache=True)
def abm_4_solver(y0, t_span, h, epsilon, extinction_threshold):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 3))
    y[0] = y0
    if n_steps == 1: return t, y
    for i in range(min(3, n_steps - 1)):
        y[i+1] = rk4_step_fast(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
    if n_steps <= 4: return t, y
    
    f0 = system_func(t[3], y[3], epsilon, extinction_threshold)
    f1 = system_func(t[2], y[2], epsilon, extinction_threshold)
    f2 = system_func(t[1], y[1], epsilon, extinction_threshold)
    f3 = system_func(t[0], y[0], epsilon, extinction_threshold)
    
    for i in range(3, n_steps - 1):
        y_p = y[i] + h/24.0 * (55.0*f0 - 59.0*f1 + 37.0*f2 - 9.0*f3)
        y[i+1] = y[i] + h/24.0 * (9.0*system_func(t[i+1], y_p, epsilon, extinction_threshold) + 19.0*f0 - 5.0*f1 + f2)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
        
        f3 = f2
        f2 = f1
        f1 = f0
        f0 = system_func(t[i+1], y[i+1], epsilon, extinction_threshold)
    return t, y

@njit(cache=True)
def gear_4_solver(y0, t_span, h, epsilon, extinction_threshold, tol=1e-7, max_iter=100):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 3))
    y[0] = y0
    if n_steps == 1: return t, y
    for i in range(min(3, n_steps - 1)):
        y[i+1] = rk4_step_fast(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
        
    I = np.eye(3)
    for i in range(3, n_steps - 1):
        y_guess = y[i].copy()
        converged = False
        for _ in range(max_iter):
            fz = system_func(t[i+1], y_guess, epsilon, extinction_threshold)
            F = y_guess - (48.0/25.0)*y[i] + (36.0/25.0)*y[i-1] - (16.0/25.0)*y[i-2] + (3.0/25.0)*y[i-3] - (12.0/25.0)*h*fz
            J_f = system_jacobian(y_guess, epsilon, extinction_threshold)
            J_F = I - (12.0/25.0 * h) * J_f
            if np.any(np.isnan(J_F)) or np.any(np.isinf(J_F)) or np.any(np.isnan(F)) or np.any(np.isinf(F)): break
            try: dy = np.linalg.solve(J_F, -F)
            except: break
            y_guess = y_guess + dy
            if np.linalg.norm(dy) < tol:
                converged = True
                break
        y[i+1] = y_guess
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
    return t, y

# SYSTEM 2 SOLVERS
@njit(cache=True)
def adams_bashfort_4_solver2(y0, t_span, h, epsilon, extinction_threshold):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 4))
    y[0] = y0
    if n_steps == 1: return t, y
    for i in range(min(3, n_steps - 1)):
        y[i+1] = rk4_step_fast2(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]

    for i in range(3, n_steps - 1):
        f0 = system2_func(t[i], y[i], epsilon, extinction_threshold)
        f1 = system2_func(t[i-1], y[i-1], epsilon, extinction_threshold)
        f2 = system2_func(t[i-2], y[i-2], epsilon, extinction_threshold)
        f3 = system2_func(t[i-3], y[i-3], epsilon, extinction_threshold)
        y[i+1] = y[i] + h/24.0 * (55.0*f0 - 59.0*f1 + 37.0*f2 - 9.0*f3)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
    return t, y

@njit(cache=True)
def adams_molton_4_solver2(y0, t_span, h, epsilon, extinction_threshold, tol=1e-7, max_iter=100):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 4))
    y[0] = y0
    if n_steps == 1: return t, y
    for i in range(min(3, n_steps - 1)):
        y[i+1] = rk4_step_fast2(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
        
    if n_steps <= 4: return t, y
    
    I = np.eye(4)
    f0 = system2_func(t[3], y[3], epsilon, extinction_threshold)
    f1 = system2_func(t[2], y[2], epsilon, extinction_threshold)
    f2 = system2_func(t[1], y[1], epsilon, extinction_threshold)
    
    for i in range(3, n_steps - 1):
        y_guess = y[i].copy()
        converged = False
        for _ in range(max_iter):
            fz = system2_func(t[i+1], y_guess, epsilon, extinction_threshold)
            F = y_guess - y[i] - (h / 24.0) * (9.0 * fz + 19.0 * f0 - 5.0 * f1 + f2)
            J_f = system2_jacobian(y_guess, epsilon, extinction_threshold)
            J_F = I - (h * 9.0 / 24.0) * J_f
            if np.any(np.isnan(J_F)) or np.any(np.isinf(J_F)) or np.any(np.isnan(F)) or np.any(np.isinf(F)): break
            try: dy = np.linalg.solve(J_F, -F)
            except: break
            y_guess = y_guess + dy
            if np.linalg.norm(dy) < tol:
                converged = True
                break
        y[i+1] = y_guess
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
        f2 = f1
        f1 = f0
        f0 = system2_func(t[i+1], y[i+1], epsilon, extinction_threshold)
    return t, y

@njit(cache=True)
def abm_4_solver2(y0, t_span, h, epsilon, extinction_threshold):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 4))
    y[0] = y0
    if n_steps == 1: return t, y
    for i in range(min(3, n_steps - 1)):
        y[i+1] = rk4_step_fast2(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
    if n_steps <= 4: return t, y
    
    f0 = system2_func(t[3], y[3], epsilon, extinction_threshold)
    f1 = system2_func(t[2], y[2], epsilon, extinction_threshold)
    f2 = system2_func(t[1], y[1], epsilon, extinction_threshold)
    f3 = system2_func(t[0], y[0], epsilon, extinction_threshold)
    
    for i in range(3, n_steps - 1):
        y_p = y[i] + h/24.0 * (55.0*f0 - 59.0*f1 + 37.0*f2 - 9.0*f3)
        y[i+1] = y[i] + h/24.0 * (9.0*system2_func(t[i+1], y_p, epsilon, extinction_threshold) + 19.0*f0 - 5.0*f1 + f2)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
        
        f3 = f2
        f2 = f1
        f1 = f0
        f0 = system2_func(t[i+1], y[i+1], epsilon, extinction_threshold)
    return t, y

@njit(cache=True)
def gear_4_solver2(y0, t_span, h, epsilon, extinction_threshold, tol=1e-7, max_iter=100):
    n_steps = int(np.ceil((t_span[1] - t_span[0]) / h)) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros((n_steps, 4))
    y[0] = y0
    if n_steps == 1: return t, y
    for i in range(min(3, n_steps - 1)):
        y[i+1] = rk4_step_fast2(t[i], y[i], h, epsilon, extinction_threshold)
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
        
    I = np.eye(4)
    for i in range(3, n_steps - 1):
        y_guess = y[i].copy()
        converged = False
        for _ in range(max_iter):
            fz = system2_func(t[i+1], y_guess, epsilon, extinction_threshold)
            F = y_guess - (48.0/25.0)*y[i] + (36.0/25.0)*y[i-1] - (16.0/25.0)*y[i-2] + (3.0/25.0)*y[i-3] - (12.0/25.0)*h*fz
            J_f = system2_jacobian(y_guess, epsilon, extinction_threshold)
            J_F = I - (12.0/25.0 * h) * J_f
            if np.any(np.isnan(J_F)) or np.any(np.isinf(J_F)) or np.any(np.isnan(F)) or np.any(np.isinf(F)): break
            try: dy = np.linalg.solve(J_F, -F)
            except: break
            y_guess = y_guess + dy
            if np.linalg.norm(dy) < tol:
                converged = True
                break
        y[i+1] = y_guess
        if np.any(np.isnan(y[i+1])) or np.any(np.isinf(y[i+1])): return t[:i+2], y[:i+2]
    return t, y
# @st.cache_data (disabled to save memory)
def run_simulation(solver_name, x0_val, y0_val, alpha20_val, Tk_val, h_val, eps_val, extinction_thresh_val):
    t_span = (0.0, float(Tk_val))
    y0 = np.array([float(x0_val), float(y0_val), float(alpha20_val)])
    
    if solver_name == "Implicit Euler (Newton, Custom)":
        return implicit_euler_newton(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Runge-Kutta 4 (Explicit, Custom)":
        return rk4_solver(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Explicit Euler (Custom)":
        return euler_explicit(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Adams-Bashforth 4 (Explicit, Custom)":
        return adams_bashfort_4_solver(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Adams-Moulton 4 (Implicit, Custom)":
        return adams_molton_4_solver(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Adams-Bashforth-Moulton 4 (Predictor-Corrector, Custom)":
        return abm_4_solver(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Gear 4 / BDF4 (Implicit, Custom)":
        return gear_4_solver(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "SciPy BDF (Stiff)":
        return scipy_solver_wrapper(y0, t_span, h_val, eps_val, extinction_thresh_val, 'BDF')
    elif solver_name == "SciPy Radau (Stiff)":
        return scipy_solver_wrapper(y0, t_span, h_val, eps_val, extinction_thresh_val, 'Radau')
    elif solver_name == "SciPy LSODA (Stiff/Non-Stiff)":
        return scipy_solver_wrapper(y0, t_span, h_val, eps_val, extinction_thresh_val, 'LSODA')
    elif solver_name == "SciPy RK45 (Explicit)":
        return scipy_solver_wrapper(y0, t_span, h_val, eps_val, extinction_thresh_val, 'RK45')
    return np.array([0.0]), np.zeros((1, 3))

# @st.cache_data (disabled to save memory)
def run_simulation2(solver_name, x0_val, y0_val, alpha10_val, alpha20_val, Tk_val, h_val, eps_val, extinction_thresh_val):
    t_span = (0.0, float(Tk_val))
    y0 = np.array([float(x0_val), float(y0_val), float(alpha10_val), float(alpha20_val)])
    
    if solver_name == "Implicit Euler (Newton, Custom)":
        return implicit_euler_newton2(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Runge-Kutta 4 (Explicit, Custom)":
        return rk4_solver2(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Explicit Euler (Custom)":
        return euler_explicit2(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Adams-Bashforth 4 (Explicit, Custom)":
        return adams_bashfort_4_solver2(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Adams-Moulton 4 (Implicit, Custom)":
        return adams_molton_4_solver2(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Adams-Bashforth-Moulton 4 (Predictor-Corrector, Custom)":
        return abm_4_solver2(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "Gear 4 / BDF4 (Implicit, Custom)":
        return gear_4_solver2(y0, t_span, h_val, eps_val, extinction_thresh_val)
    elif solver_name == "SciPy BDF (Stiff)":
        return scipy_solver_wrapper2(y0, t_span, h_val, eps_val, extinction_thresh_val, 'BDF')
    elif solver_name == "SciPy Radau (Stiff)":
        return scipy_solver_wrapper2(y0, t_span, h_val, eps_val, extinction_thresh_val, 'Radau')
    elif solver_name == "SciPy LSODA (Stiff/Non-Stiff)":
        return scipy_solver_wrapper2(y0, t_span, h_val, eps_val, extinction_thresh_val, 'LSODA')
    elif solver_name == "SciPy RK45 (Explicit)":
        return scipy_solver_wrapper2(y0, t_span, h_val, eps_val, extinction_thresh_val, 'RK45')
    return np.array([0.0]), np.zeros((1, 4))

# Setup integration run and state vectors
t_span = (0.0, float(Tk))
start_time = time.time()

with st.spinner("Решение системы (интегрирование)..."):
    if model_selection == "Система 1 (3D Адаптивная)":
        y0_vec = np.array([float(x0), float(y0), float(alpha20)])
        t, y = run_simulation(solver_name, x0, y0, alpha20, Tk, h, eps, extinction_thresh)
    else:
        y0_vec = np.array([float(x0), float(y0), float(alpha10), float(alpha20)])
        t, y = run_simulation2(solver_name, x0, y0, alpha10, alpha20, Tk, h, eps, extinction_thresh)

execution_time = time.time() - start_time

# Display simulation stats
st.markdown("### 📊 Simulation Summary")
stat_col1, stat_col2, stat_col3 = st.columns(3)

# Check for solver explosion/instability
is_stable = True
error_reason = ""
if len(t) < 2:
    is_stable = False
    error_reason = "Solver failed immediately due to invalid inputs."
elif np.any(np.isnan(y[-1])) or np.any(np.isinf(y[-1])):
    is_stable = False
    error_reason = "Solver exploded (Inf/NaN). Typical for explicit solvers on stiff systems!"
elif t[-1] < t_span[1] - h * 1.5:
    is_stable = False
    error_reason = f"Solver exploded or stopped at t = {t[-1]:.2f}. Instability encountered."

with stat_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Status</div>
        <div class="metric-val" style="color: {'#10b981' if is_stable else '#ef4444'};">
            {'Stable' if is_stable else 'Failed'}
        </div>
    </div>
    """, unsafe_allow_html=True)
with stat_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Actual End Time / Simulation Steps</div>
        <div class="metric-val">{t[-1]:.2f} / {len(t):,}</div>
    </div>
    """, unsafe_allow_html=True)
with stat_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Computation Time</div>
        <div class="metric-val">{execution_time*1000:.2f} ms</div>
    </div>
    """, unsafe_allow_html=True)

if not is_stable:
    st.error(f"⚠️ **Simulation unstable:** {error_reason}")

# ----------------------------------------------------
# 5. VISUALIZATION LAYOUT (SIDE-BY-SIDE)
# ----------------------------------------------------
if len(t) > 1:
    col_left, col_right = st.columns([1, 1])
    
    # SYSTEM 1 LAYOUT
    if model_selection == "Система 1 (3D Адаптивная)":
        # LEFT COLUMN: Trajectories
        with col_left:
            st.subheader("📉 Траектории (Временные ряды)")
            
            fig_ts = make_subplots(rows=3, cols=1, 
                                   shared_xaxes=True, 
                                   vertical_spacing=0.08,
                                   subplot_titles=("Популяция жертвы: x(t)", "Популяция хищника: y(t)", "Эффективность поиска: α2(t)"))
            
            fig_ts.add_trace(go.Scatter(x=t, y=y[:, 0], name="Жертва (x)", line=dict(color="#38bdf8", width=2)), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=t, y=y[:, 1], name="Хищник (y)", line=dict(color="#fb923c", width=2)), row=2, col=1)
            fig_ts.add_trace(go.Scatter(x=t, y=y[:, 2], name="Адаптивность (α2)", line=dict(color="#a78bfa", width=2)), row=3, col=1)
            
            fig_ts.update_layout(
                height=720, 
                showlegend=False, 
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig_ts.update_xaxes(title_text="Время (t)", row=3, col=1, gridcolor="#e2e8f0")
            fig_ts.update_yaxes(gridcolor="#e2e8f0")
            st.plotly_chart(fig_ts, use_container_width=True)
            
        # LEFT COLUMN: Stiffness & Phase portraits
        with col_left:
            st.subheader("📈 Анализ жесткости и собственных значений")
            decimation = max(1, len(t) // 1000)
            t_stiff = t[::decimation]
            y_stiff = y[::decimation]
            
            eigvals, stiffness = compute_stiffness_and_eigenvalues(t_stiff, y_stiff, eps, extinction_thresh)
            
            # Plot 1: Stiffness ratio
            fig_stiff = go.Figure()
            fig_stiff.add_trace(go.Scatter(x=t_stiff, y=stiffness, name="Stiffness Ratio L(t)", line=dict(color="#f43f5e", width=2.5)))
            fig_stiff.update_layout(
                template="plotly_white",
                title="Stiffness Ratio over Time L(t) (Log Scale)",
                xaxis_title="Time t",
                yaxis_title="L(t)",
                yaxis_type="log",
                height=320,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0")
            )
            st.plotly_chart(fig_stiff, use_container_width=True)
            
            st.markdown(r"""
            Система является жесткой, если собственные значения Якобиана $\lambda_i$ сильно различаются по масштабу. 
            Коэффициент жесткости: $L(t) = \frac{\max_i |\text{Re}(\lambda_i)|}{\min_i |\text{Re}(\lambda_i)|}$.
            """)
            
            # Plot 2: Real parts of Eigenvalues
            fig_eig = go.Figure()
            fig_eig.add_trace(go.Scatter(x=t_stiff, y=eigvals[:, 0], name="Re(λ1)", line=dict(color="#22c55e", width=2)))
            fig_eig.add_trace(go.Scatter(x=t_stiff, y=eigvals[:, 1], name="Re(λ2)", line=dict(color="#eab308", width=2)))
            fig_eig.add_trace(go.Scatter(x=t_stiff, y=eigvals[:, 2], name="Re(λ3)", line=dict(color="#a855f7", width=2)))
            
            st.write("Вещественные части собственных значений: $\\text{Re}(\\lambda_i)$")
            scale_type = st.radio("Масштаб оси Y:", ["Линейный", "Логарифмический (по модулю)"], horizontal=True, key="scale_s1")
            
            if scale_type == "Логарифмический (по модулю)":
                fig_eig_log = go.Figure()
                fig_eig_log.add_trace(go.Scatter(x=t_stiff, y=np.abs(eigvals[:, 0]), name="|Re(λ1)|", line=dict(color="#22c55e", width=2)))
                fig_eig_log.add_trace(go.Scatter(x=t_stiff, y=np.abs(eigvals[:, 1]), name="|Re(λ2)|", line=dict(color="#eab308", width=2)))
                fig_eig_log.add_trace(go.Scatter(x=t_stiff, y=np.abs(eigvals[:, 2]), name="|Re(λ3)|", line=dict(color="#a855f7", width=2)))
                fig_eig_log.update_layout(
                    template="plotly_white", xaxis_title="Время t", yaxis_title="Абс. вещественная часть |Re(λ)|", yaxis_type="log", height=320,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0")
                )
                st.plotly_chart(fig_eig_log, use_container_width=True)
            else:
                fig_eig.update_layout(
                    template="plotly_white", xaxis_title="Время t", yaxis_title="Вещественная часть Re(λ)", height=320,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0")
                )
                st.plotly_chart(fig_eig, use_container_width=True)
                
            # Metrics
            valid_stiff = stiffness[~np.isnan(stiffness)]
            if len(valid_stiff) > 0:
                stiff_col1, stiff_col2 = st.columns(2)
                with stiff_col1:
                    st.metric("Средний коэф. жесткости", f"{np.mean(valid_stiff):.2e}")
                with stiff_col2:
                    st.metric("Макс. коэф. жесткости", f"{np.max(valid_stiff):.2e}")
                    
        with col_right:
            # Phase Space portraits (colored by time)
            st.subheader("🌀 Фазовые Портреты (По времени)")
            port_col1 = st.container()
            port_col2 = st.container()
            
            with port_col1:
                st.write("2D Фазовый портрет ($x$ vs $y$)")
                fig_2d = go.Figure()
                fig_2d.add_trace(go.Scatter(
                    x=y[:, 0], y=y[:, 1], mode='markers+lines',
                    marker=dict(
                        color=t,
                        colorscale='Viridis',
                        size=3,
                        showscale=True,
                        colorbar=dict(title="Время t", thickness=15, x=1.05)
                    ),
                    line=dict(color='rgba(0, 0, 0, 0.15)', width=1),
                    hoverinfo='text',
                    text=[f"t={ti:.1f}<br>x={xi:.3f}<br>y={yi:.3f}" for ti, xi, yi in zip(t, y[:, 0], y[:, 1])]
                ))
                fig_2d.update_layout(
                    template="plotly_white", xaxis_title="Prey x", yaxis_title="Predator y", height=350,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0")
                )
                st.plotly_chart(fig_2d, use_container_width=True)
                
            with port_col2:
                st.write("3D Фазовый портрет ($x$, $y$, $\\alpha_2$)")
                fig_3d = go.Figure()
                fig_3d.add_trace(go.Scatter3d(
                    x=y[:, 0], y=y[:, 1], z=y[:, 2], mode='lines',
                    line=dict(
                        color=t,
                        colorscale='Viridis',
                        width=4,
                        showscale=True,
                        colorbar=dict(title="Время t", thickness=15, x=-0.2)
                    ),
                    hoverinfo='text',
                    text=[f"t={ti:.1f}<br>x={xi:.3f}<br>y={yi:.3f}<br>α2={ai:.4f}" for ti, xi, yi, ai in zip(t, y[:, 0], y[:, 1], y[:, 2])]
                ))
                fig_3d.update_layout(
                    template="plotly_white",
                    scene=dict(
                        xaxis_title="Жертва x", yaxis_title="Хищник y", zaxis_title="Адаптивность α2",
                        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"), zaxis=dict(gridcolor="#e2e8f0")
                    ),
                    height=350, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
    # SYSTEM 2 LAYOUT
    else:
        # LEFT COLUMN: Trajectories (System 2 has 4 dimensions)
        with col_left:
            st.subheader("📉 Траектории (Временные ряды)")
            
            fig_ts = make_subplots(rows=4, cols=1, 
                                   shared_xaxes=True, 
                                   vertical_spacing=0.06,
                                   subplot_titles=("Популяция жертвы: x(t)", "Популяция хищника: y(t)", "Адаптивный признак α1(t)", "Адаптивный признак α2(t)"))
            
            fig_ts.add_trace(go.Scatter(x=t, y=y[:, 0], name="Жертва (x)", line=dict(color="#38bdf8", width=2)), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=t, y=y[:, 1], name="Хищник (y)", line=dict(color="#fb923c", width=2)), row=2, col=1)
            fig_ts.add_trace(go.Scatter(x=t, y=y[:, 2], name="Адаптивность (α1)", line=dict(color="#22c55e", width=2)), row=3, col=1)
            fig_ts.add_trace(go.Scatter(x=t, y=y[:, 3], name="Адаптивность (α2)", line=dict(color="#a78bfa", width=2)), row=4, col=1)
            
            fig_ts.update_layout(
                    height=780, 
                    showlegend=False, 
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            fig_ts.update_xaxes(title_text="Время (t)", row=4, col=1, gridcolor="#e2e8f0")
            fig_ts.update_yaxes(gridcolor="#e2e8f0")
            st.plotly_chart(fig_ts, use_container_width=True)
            
        # LEFT COLUMN: Stiffness & Phase portraits (System 2)
        with col_left:
            st.subheader("📈 Анализ жесткости и собственных значений")
            decimation = max(1, len(t) // 1000)
            t_stiff = t[::decimation]
            y_stiff = y[::decimation]
            
            eigvals, stiffness = compute_stiffness_and_eigenvalues2(t_stiff, y_stiff, eps, extinction_thresh)
            
            # Plot 1: Stiffness ratio
            fig_stiff = go.Figure()
            fig_stiff.add_trace(go.Scatter(x=t_stiff, y=stiffness, name="Жесткость S(t)", line=dict(color="#f43f5e", width=2.5)))
            fig_stiff.update_layout(
                template="plotly_white",
                title="Жесткость S(t) от времени (Лог. шкала)",
                xaxis_title="Время t",
                yaxis_title="S(t)",
                yaxis_type="log",
                height=320,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0")
            )
            st.plotly_chart(fig_stiff, use_container_width=True)
            
            st.markdown(r"""
            Система является жесткой, если собственные значения Якобиана $\lambda_i$ сильно различаются по масштабу. 
            Коэффициент жесткости: $S(t) = \frac{\max_i |\text{Re}(\lambda_i)|}{\min_i |\text{Re}(\lambda_i)|}$.
            """)
            
            # Plot 2: Real parts of Eigenvalues (4 modes)
            fig_eig = go.Figure()
            fig_eig.add_trace(go.Scatter(x=t_stiff, y=eigvals[:, 0], name="Re(λ1)", line=dict(color="#22c55e", width=2)))
            fig_eig.add_trace(go.Scatter(x=t_stiff, y=eigvals[:, 1], name="Re(λ2)", line=dict(color="#eab308", width=2)))
            fig_eig.add_trace(go.Scatter(x=t_stiff, y=eigvals[:, 2], name="Re(λ3)", line=dict(color="#a855f7", width=2)))
            fig_eig.add_trace(go.Scatter(x=t_stiff, y=eigvals[:, 3], name="Re(λ4)", line=dict(color="#ec4899", width=2)))
            
            st.write("Вещественные части собственных значений: $\\text{Re}(\\lambda_i)$")
            scale_type = st.radio("Масштаб оси Y:", ["Линейный", "Логарифмический (по модулю)"], horizontal=True, key="scale_s2")
            
            if scale_type == "Логарифмический (по модулю)":
                fig_eig_log = go.Figure()
                fig_eig_log.add_trace(go.Scatter(x=t_stiff, y=np.abs(eigvals[:, 0]), name="|Re(λ1)|", line=dict(color="#22c55e", width=2)))
                fig_eig_log.add_trace(go.Scatter(x=t_stiff, y=np.abs(eigvals[:, 1]), name="|Re(λ2)|", line=dict(color="#eab308", width=2)))
                fig_eig_log.add_trace(go.Scatter(x=t_stiff, y=np.abs(eigvals[:, 2]), name="|Re(λ3)|", line=dict(color="#a855f7", width=2)))
                fig_eig_log.add_trace(go.Scatter(x=t_stiff, y=np.abs(eigvals[:, 3]), name="|Re(λ4)|", line=dict(color="#ec4899", width=2)))
                fig_eig_log.update_layout(
                    template="plotly_white", xaxis_title="Время t", yaxis_title="Абс. вещественная часть |Re(λ)|", yaxis_type="log", height=320,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0")
                )
                st.plotly_chart(fig_eig_log, use_container_width=True)
            else:
                fig_eig.update_layout(
                    template="plotly_white", xaxis_title="Время t", yaxis_title="Вещественная часть Re(λ)", height=320,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0")
                )
                st.plotly_chart(fig_eig, use_container_width=True)
                
            # Metrics
            valid_stiff = stiffness[~np.isnan(stiffness)]
            if len(valid_stiff) > 0:
                stiff_col1, stiff_col2 = st.columns(2)
                with stiff_col1:
                    st.metric("Average Stiffness Ratio", f"{np.mean(valid_stiff):.2e}")
                with stiff_col2:
                    st.metric("Maximum Stiffness Ratio", f"{np.max(valid_stiff):.2e}")
                    
        with col_right:
            # Phase Space portraits (System 2, colored by time)
            st.subheader("🌀 Фазовые Портреты (По времени)")
            port_col1 = st.container()
            port_col2 = st.container()
            
            with port_col1:
                st.write("2D Фазовый портрет ($x$ от $y$)")
                fig_2d = go.Figure()
                fig_2d.add_trace(go.Scatter(
                    x=y[:, 0], y=y[:, 1], mode='markers+lines',
                    marker=dict(
                        color=t,
                        colorscale='Viridis',
                        size=3,
                        showscale=True,
                        colorbar=dict(title="Время t", thickness=15, x=1.05)
                    ),
                    line=dict(color='rgba(0, 0, 0, 0.15)', width=1),
                    hoverinfo='text',
                    text=[f"t={ti:.1f}<br>x={xi:.3f}<br>y={yi:.3f}" for ti, xi, yi in zip(t, y[:, 0], y[:, 1])]
                ))
                fig_2d.update_layout(
                    template="plotly_white", xaxis_title="Жертва x", yaxis_title="Хищник y", height=350,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0")
                )
                st.plotly_chart(fig_2d, use_container_width=True)
                
            with port_col2:
                z_var = st.radio("Ось Z для 3D портрета:", ["alpha1 (α1)", "alpha2 (α2)"], horizontal=True, key="z_var_s2")
                z_data = y[:, 2] if "alpha1" in z_var else y[:, 3]
                z_label = "Адаптивность α1" if "alpha1" in z_var else "Адаптивность α2"
                
                fig_3d = go.Figure()
                fig_3d.add_trace(go.Scatter3d(
                    x=y[:, 0], y=y[:, 1], z=z_data, mode='lines',
                    line=dict(
                        color=t,
                        colorscale='Viridis',
                        width=4,
                        showscale=True,
                        colorbar=dict(title="Время t", thickness=15, x=-0.2)
                    ),
                    hoverinfo='text',
                    text=[f"t={ti:.1f}<br>x={xi:.3f}<br>y={yi:.3f}<br>α1={a1i:.4f}<br>α2={a2i:.4f}" for ti, xi, yi, a1i, a2i in zip(t, y[:, 0], y[:, 1], y[:, 2], y[:, 3])]
                ))
                fig_3d.update_layout(
                    template="plotly_white",
                    scene=dict(
                        xaxis_title="Жертва x", yaxis_title="Хищник y", zaxis_title=z_label,
                        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"), zaxis=dict(gridcolor="#e2e8f0")
                    ),
                    height=350, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.warning("Insufficient data to plot. Simulation did not run successfully.")

# ----------------------------------------------------
# 6. SOLVER COMPARISON BENCHMARK
# ----------------------------------------------------
st.markdown("---")
with st.expander("🔀 Solver Stability and Speed Comparison", expanded=False):
    st.subheader("Stability and Speed Comparison of Solvers")
    st.markdown("""
    Compare how different numerical methods handle this stiff predator-prey system. 
    Explicit methods (Euler, RK4, RK45) struggle or blow up on stiff systems unless the step size $h$ is extremely small.
    Implicit methods (Implicit Euler, BDF, Radau, LSODA) can handle much larger step sizes.
    """)
    
    st.write("Click the button below to run a batch benchmark using the current settings.")
    
    if st.button("🚀 Run Solver Benchmark"):
        if model_selection == "Система 1 (3D Адаптивная)":
            solvers_to_run = {
                "Explicit Euler": lambda: euler_explicit(y0_vec, t_span, h, eps, extinction_thresh),
                "RK4 (Explicit)": lambda: rk4_solver(y0_vec, t_span, h, eps, extinction_thresh),
                "Implicit Euler (Newton)": lambda: implicit_euler_newton(y0_vec, t_span, h, eps, extinction_thresh),
                "SciPy RK45 (Explicit)": lambda: scipy_solver_wrapper(y0_vec, t_span, h, eps, extinction_thresh, 'RK45'),
                "SciPy BDF (Implicit)": lambda: scipy_solver_wrapper(y0_vec, t_span, h, eps, extinction_thresh, 'BDF'),
                "SciPy Radau (Implicit)": lambda: scipy_solver_wrapper(y0_vec, t_span, h, eps, extinction_thresh, 'Radau'),
                "SciPy LSODA (Auto)": lambda: scipy_solver_wrapper(y0_vec, t_span, h, eps, extinction_thresh, 'LSODA')
            }
        else:
            solvers_to_run = {
                "Explicit Euler": lambda: euler_explicit2(y0_vec, t_span, h, eps, extinction_thresh),
                "RK4 (Explicit)": lambda: rk4_solver2(y0_vec, t_span, h, eps, extinction_thresh),
                "Implicit Euler (Newton)": lambda: implicit_euler_newton2(y0_vec, t_span, h, eps, extinction_thresh),
                "SciPy RK45 (Explicit)": lambda: scipy_solver_wrapper2(y0_vec, t_span, h, eps, extinction_thresh, 'RK45'),
                "SciPy BDF (Implicit)": lambda: scipy_solver_wrapper2(y0_vec, t_span, h, eps, extinction_thresh, 'BDF'),
                "SciPy Radau (Implicit)": lambda: scipy_solver_wrapper2(y0_vec, t_span, h, eps, extinction_thresh, 'Radau'),
                "SciPy LSODA (Auto)": lambda: scipy_solver_wrapper2(y0_vec, t_span, h, eps, extinction_thresh, 'LSODA')
            }
            
        results = []
        
        for name, solver_fn in solvers_to_run.items():
            t_start = time.time()
            try:
                t_res, y_res = solver_fn()
                t_elapsed = time.time() - t_start
                
                # Check status
                if len(t_res) < 2:
                    status = "Failed"
                    reason = "Immediate failure"
                elif np.any(np.isnan(y_res[-1])) or np.any(np.isinf(y_res[-1])):
                    status = "Exploded"
                    reason = "NaN/Inf value"
                elif t_res[-1] < t_span[1] - h * 1.5:
                    status = "Incomplete"
                    reason = f"Stopped at t = {t_res[-1]:.1f}"
                else:
                    status = "Successful"
                    reason = "Completed full span"
                
                results.append({
                    "Solver Name": name,
                    "Status": status,
                    "Simulated Time": f"{t_res[-1]:.2f} / {t_span[1]:.2f}",
                    "Actual Steps": len(t_res),
                    "Execution Time (ms)": round(t_elapsed * 1000, 3),
                    "Info": reason
                })
            except Exception as e:
                results.append({
                    "Solver Name": name,
                    "Status": "Error",
                    "Simulated Time": "0.00",
                    "Actual Steps": 0,
                    "Execution Time (ms)": 0,
                    "Info": str(e)
                })
                
        df_results = pd.DataFrame(results)
        
        # Color styling function for status
        def style_status(val):
            if val == "Successful":
                color = "#10b981"
            elif val in ["Exploded", "Failed"]:
                color = "#ef4444"
            else:
                color = "#eab308"
            return f"color: {color}; font-weight: bold;"
            
        st.dataframe(
            df_results.style.map(style_status, subset=["Status"]),
            use_container_width=True
        )
        
        st.info("💡 Notice that BDF, Radau, LSODA, and our custom Newton-Implicit Euler complete successfully, whereas explicit solvers often explode or stop early unless the step size $h$ is very small.")

