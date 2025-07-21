import streamlit as st
import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Analysis function
# num_elem = 200
def analyze_beam(L, supports, loads, udls, E, I, num_elem):
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    node_coords = np.linspace(0, L, num_elem + 1)
    for i, x in enumerate(node_coords):
        ops.node(i+1, x, 0)
    for n in supports:
        ops.fix(n, 1, 1, 0)
    ops.geomTransf('Linear', 1)
    for i in range(num_elem):
        ops.element('elasticBeamColumn', i+1, i+1, i+2, 1, E, I, 1)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for (n, P) in loads:
        ops.load(n, 0, -P, 0)
    for (start, end, q) in udls:
        for i in range(len(node_coords)-1):
            xi, xj = node_coords[i], node_coords[i+1]
            if xi >= start and xj <= end:
                Le = xj - xi
                P_eq = q * Le / 2
                ops.load(i+1, 0, -P_eq, 0)
                ops.load(i+2, 0, -P_eq, 0)
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)

    x = node_coords
    D = np.array([ops.nodeDisp(i+1, 2) for i in range(len(x))])

    # Extract internal shear/moment from elements
    V, M, x_mid = [], [], []
    for i in range(num_elem):
        ele_forces = ops.eleForce(i+1)
        V.append(-ele_forces[1])     # Shear
        M.append(ele_forces[2])      # Moment
        x_mid.append((x[i] + x[i+1]) / 2)

    return np.array(x_mid), np.array(M), np.array(V), D

st.title('Simple Beam Analysis with OpenSeesPy')

# ↓ NEW ↓
num_elem = st.sidebar.number_input(
    'Number of Elements',
    min_value=1,
    value=200,
    step=1
)
# --- Length Unit Handling ---
if 'length_unit_prev' not in st.session_state:
    st.session_state.length_unit_prev = 'm'

length_unit = st.sidebar.selectbox(
    'Length Unit', ['m', 'mm'],
    index=['m','mm'].index(st.session_state.length_unit_prev)
)

if length_unit != st.session_state.length_unit_prev:
    old = st.session_state.length_unit_prev
    old_f = 1.0 if old=='m' else 1000.0
    new_f = 1.0 if length_unit=='m' else 1000.0
    r = new_f / old_f

    # Scale beam length
    st.session_state['L_input'] = st.session_state.get('L_input',10.0) * r

    # Scale supports & loads positions
    for key in list(st.session_state.keys()):
        if key.startswith(('sup_var_','load_var_','udl_start_var_','udl_end_var_')):
            st.session_state[key] *= r

    # Scale UDL intensities (N/m ↔ N/mm)
    for key in list(st.session_state.keys()):
        if key.startswith('udl_q_var_'):
            st.session_state[key] /= r

    # Convert E and custom I inputs
    if old=='m' and length_unit=='mm':
        st.session_state['E_input'] = st.session_state.get('E_input',2.1e11) / 1e6   # Pa → MPa
        st.session_state['I_input'] = st.session_state.get('I_input',8.33e-6) * 1e12 # m⁴ → mm⁴
    elif old=='mm' and length_unit=='m':
        st.session_state['E_input'] = st.session_state.get('E_input',2.1e5) * 1e6    # MPa → Pa
        st.session_state['I_input'] = st.session_state.get('I_input',8.33e6) / 1e12  # mm⁴ → m⁴

    # Scale ALL section dimensions
    section_keys = [
        'rect_b','rect_h',
        'hrect_B','hrect_H','hrect_b','hrect_h',
        'circle_d',
        'outer_diameter','inner_diameter',
        'ibeam_B','ibeam_H','ibeam_b','ibeam_h',
        'tbeam_b','tbeam_h',
        'lbeam_b','lbeam_h',
    ]
    for key in section_keys:
        if key in st.session_state:
            st.session_state[key] *= r

    st.session_state.length_unit_prev = length_unit

factor = 1.0 if length_unit=='m' else 1000.0

# --- Section Defaults ---
def init_section_defaults(length_unit):
    if 'rect_b' not in st.session_state:
        st.session_state['rect_b'] = 0.3 if length_unit=='m' else 300.0
    if 'rect_h' not in st.session_state:
        st.session_state['rect_h'] = 0.6 if length_unit=='m' else 600.0

    if 'hrect_B' not in st.session_state:
        st.session_state['hrect_B'] = 0.4 if length_unit=='m' else 400.0
    if 'hrect_H' not in st.session_state:
        st.session_state['hrect_H'] = 0.6 if length_unit=='m' else 600.0
    if 'hrect_b' not in st.session_state:
        st.session_state['hrect_b'] = 0.2 if length_unit=='m' else 200.0
    if 'hrect_h' not in st.session_state:
        st.session_state['hrect_h'] = 0.3 if length_unit=='m' else 300.0

    if 'circle_d' not in st.session_state:
        st.session_state['circle_d'] = 0.3 if length_unit=='m' else 300.0

    if 'outer_diameter' not in st.session_state:
        st.session_state['outer_diameter'] = 0.4 if length_unit=='m' else 400.0
    if 'inner_diameter' not in st.session_state:
        st.session_state['inner_diameter'] = 0.2 if length_unit=='m' else 200.0

    if 'ibeam_B' not in st.session_state:
        st.session_state['ibeam_B'] = 0.3 if length_unit=='m' else 300.0
    if 'ibeam_H' not in st.session_state:
        st.session_state['ibeam_H'] = 0.6 if length_unit=='m' else 600.0
    if 'ibeam_b' not in st.session_state:
        st.session_state['ibeam_b'] = 0.1 if length_unit=='m' else 100.0
    if 'ibeam_h' not in st.session_state:
        st.session_state['ibeam_h'] = 0.3 if length_unit=='m' else 300.0

    if 'tbeam_b' not in st.session_state:
        st.session_state['tbeam_b'] = 0.3 if length_unit=='m' else 300.0
    if 'tbeam_h' not in st.session_state:
        st.session_state['tbeam_h'] = 0.4 if length_unit=='m' else 400.0

    if 'lbeam_b' not in st.session_state:
        st.session_state['lbeam_b'] = 0.3 if length_unit=='m' else 300.0
    if 'lbeam_h' not in st.session_state:
        st.session_state['lbeam_h'] = 0.3 if length_unit=='m' else 300.0

init_section_defaults(length_unit)

# --- Inputs ---
L_input = st.number_input(
    f'Beam Length ({length_unit})',
    min_value=0.1,
    value=st.session_state.get('L_input',10.0),
    format='%.3f',
    key='L_input'
)
L = L_input / factor

E_unit = 'Pa' if length_unit=='m' else 'MPa'
E_input = st.number_input(
    f'Elastic Modulus E ({E_unit})',
    value=st.session_state.get('E_input',2.1e11 if length_unit=='m' else 2.1e5),
    format='%.3e',
    key='E_input'
)
E = E_input * (1.0 if length_unit=='m' else 1e6)

# --- Moment of Inertia Section ---
I_val = None
section_options = {
    'Rectangle': lambda b,h: b*h**3/12,
    'Hollow Rectangle': lambda B,H,b,h: (B*H**3 - b*h**3)/12,
    'Circle': lambda d: np.pi*d**4/64,
    'Hollow Circle': lambda D,d: np.pi*(D**4-d**4)/64,
    'I-Beam': lambda B,H,b,h: (B*H**3 - b*h**3)/12,
    'T-Beam': lambda b,h: b*h**3/12,
    'L-Beam (approx)': lambda b,h: b*h**3/12,
    'Custom': None
}
section = st.sidebar.selectbox("Section Type for I", list(section_options.keys()), key="sect_type")

formulas = {
    'Rectangle': r'$I = \frac{b\,h^3}{12}$',
    'Hollow Rectangle': r'$I = \frac{B\,H^3 - b\,h^3}{12}$',
    'Circle': r'$I = \frac{\pi d^4}{64}$',
    'Hollow Circle': r'$I = \frac{\pi\,(D^4 - d^4)}{64}$',
    'I-Beam': r'$I = \frac{B\,H^3 - b\,h^3}{12}$',
    'T-Beam': r'$I = \frac{b\,h^3}{12}$',
    'L-Beam (approx)': r'$I = \frac{b\,h^3}{12}$',
    'Custom': ''
}

if section != 'Custom':
    st.sidebar.markdown(f"**Formula :** {formulas[section]}")
    st.sidebar.markdown("**Section Dimensions**")

# Rectangle
if section == 'Rectangle':
    b = st.sidebar.number_input(
        f"B ({length_unit})", value=st.session_state['rect_b'], key='rect_b'
    )
    h = st.sidebar.number_input(
        f"H ({length_unit})", value=st.session_state['rect_h'], key='rect_h'
    )
    I_val = section_options[section](b,h)

# Hollow Rectangle
elif section == 'Hollow Rectangle':
    B = st.sidebar.number_input(
        f"Outer width B ({length_unit})", value=st.session_state['hrect_B'], key='hrect_B'
    )
    H = st.sidebar.number_input(
        f"Outer height H ({length_unit})", value=st.session_state['hrect_H'], key='hrect_H'
    )
    b = st.sidebar.number_input(
        f"Inner width b ({length_unit})", value=st.session_state['hrect_b'], key='hrect_b'
    )
    h = st.sidebar.number_input(
        f"Inner height h ({length_unit})", value=st.session_state['hrect_h'], key='hrect_h'
    )
    I_val = section_options[section](B,H,b,h)

# Circle
elif section == 'Circle':
    d = st.sidebar.number_input(
        f"Diameter d ({length_unit})", value=st.session_state['circle_d'], key='circle_d'
    )
    I_val = section_options[section](d)

# Hollow Circle
elif section == 'Hollow Circle':
    D = st.sidebar.number_input(
        f"Outer diameter D ({length_unit})", value=st.session_state['outer_diameter'], key='outer_diameter'
    )
    d = st.sidebar.number_input(
        f"Inner diameter d ({length_unit})", value=st.session_state['inner_diameter'], key='inner_diameter'
    )
    I_val = section_options[section](D,d)

# I-Beam
elif section == 'I-Beam':
    B = st.sidebar.number_input(
        f"Flange width B ({length_unit})", value=st.session_state['ibeam_B'], key='ibeam_B'
    )
    H = st.sidebar.number_input(
        f"Total depth H ({length_unit})", value=st.session_state['ibeam_H'], key='ibeam_H'
    )
    b = st.sidebar.number_input(
        f"Web thickness b ({length_unit})", value=st.session_state['ibeam_b'], key='ibeam_b'
    )
    h = st.sidebar.number_input(
        f"Web height h ({length_unit})", value=st.session_state['ibeam_h'], key='ibeam_h'
    )
    I_val = section_options[section](B,H,b,h)

# T-Beam
elif section == 'T-Beam':
    b = st.sidebar.number_input(
        f"Flange width b ({length_unit})", value=st.session_state['tbeam_b'], key='tbeam_b'
    )
    h = st.sidebar.number_input(
        f"Total height h ({length_unit})", value=st.session_state['tbeam_h'], key='tbeam_h'
    )
    I_val = section_options[section](b,h)

# L-Beam
elif section == 'L-Beam (approx)':
    b = st.sidebar.number_input(
        f"Leg width b ({length_unit})", value=st.session_state['lbeam_b'], key='lbeam_b'
    )
    h = st.sidebar.number_input(
        f"Leg height h ({length_unit})", value=st.session_state['lbeam_h'], key='lbeam_h'
    )
    I_val = section_options[section](b,h)

# Custom
else:
    I_unit = 'm⁴' if length_unit=='m' else 'mm⁴'
    I_input = st.number_input(
        f"Moment of Inertia I ({I_unit})",
        value=st.session_state.get('I_input',8.33e-6 if length_unit=='m' else 8.33e6),
        format='%.3e',
        key='I_input'
    )
    I_val = I_input * (1.0 if length_unit=='m' else 1e-12)

# --- Show calculated I in sidebar ---
if I_val is not None:
    I_unit_display = 'm⁴' if length_unit=='m' else 'mm⁴'
    st.sidebar.markdown(f"**Calculated I:** {I_val:.3e} {I_unit_display}")

# Final check and assignment of I
if I_val is not None:
    I = I_val
else:
    st.error("Moment of inertia could not be calculated. Please check section inputs.")
    st.stop()

# --- (rest of supports, loads, plotting, Run Analysis) ---
# [Copy the remainder of your existing Beam plotting and Run Analysis code here.]



#Sidebar: Supports & Loads
st.sidebar.header('Supports & Loads')
def init_var(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

# Supports
n_sup = st.sidebar.number_input(
    'Number of Supports', 1, 5,
    value=st.session_state.get('n_sup', 2),
    key='n_sup'
)
support_positions = []
for i in range(n_sup):
    default_pos = (i/(n_sup-1) if n_sup>1 else 0.0) * L_input
    init_var(f'sup_var_{i}', default_pos)
    col1, col2 = st.sidebar.columns([3,1])
    with col1:
        st.slider(
            f'Support {i+1} Pos ({length_unit})',
            0.0, L_input,
            value=st.session_state[f'sup_var_{i}'],
            key=f'sup_sl_{i}',
            on_change=lambda idx=i: st.session_state.update({f'sup_var_{idx}': st.session_state[f'sup_sl_{idx}']})
        )
    with col2:
        st.number_input(
            label='Pos',
            min_value=0.0,
            max_value=L_input,
            value=st.session_state[f'sup_var_{i}'],
            key=f'sup_in_{i}',
            format='%.3f',
            label_visibility='collapsed',
            on_change=lambda idx=i: st.session_state.update({f'sup_var_{idx}': st.session_state[f'sup_in_{idx}']})
        )
    support_positions.append(st.session_state[f'sup_var_{i}']/factor)

# Point Loads
st.sidebar.markdown('**Point Load Values (N)**')
n_loads = st.sidebar.number_input(
    'Number of Point Loads', 0, 10,
    value=st.session_state.get('n_loads', 1),
    key='n_loads'
)
load_positions, load_mags = [], []
for i in range(n_loads):
    init_var(f'load_var_{i}', L_input/2)
    col1, col2, col3 = st.sidebar.columns([3,1,1])
    with col1:
        st.slider(
            f'Load {i+1} Pos ({length_unit})',
            0.0, L_input,
            value=st.session_state[f'load_var_{i}'],
            key=f'load_sl_{i}',
            on_change=lambda idx=i: st.session_state.update({f'load_var_{idx}': st.session_state[f'load_sl_{idx}']})
        )
    with col2:
        st.number_input(
            label='Pos',
            min_value=0.0,
            max_value=L_input,
            value=st.session_state[f'load_var_{i}'],
            key=f'load_in_{i}',
            format='%.3f',
            label_visibility='collapsed',
            on_change=lambda idx=i: st.session_state.update({f'load_var_{idx}': st.session_state[f'load_in_{idx}']})
        )
    st.sidebar.markdown(f"Load {i+1} Magnitude (N):")
    mag = st.sidebar.number_input(
        label='Mag (N)',
        min_value=0.0,
        value=st.session_state.get(f'load_mag_{i}', 1000.0),
        format='%.1f',
        key=f'load_mag_{i}',
        label_visibility='collapsed'
    )
    load_positions.append(st.session_state[f'load_var_{i}']/factor)
    load_mags.append(mag)

# UDLs
st.sidebar.markdown(f'**UDL Intensities (N/{length_unit})**')
n_udl = st.sidebar.number_input(
    'Number of UDLs', 0, 5,
    value=st.session_state.get('n_udl', 0),
    key='n_udl'
)
udl_list = []
for i in range(n_udl):
    st.sidebar.markdown(f'**UDL {i+1}**')
    init_var(f'udl_start_var_{i}', 0.0)
    col1, col2 = st.sidebar.columns([3,1])
    with col1:
        st.slider(
            f'UDL {i+1} Start ({length_unit})',
            0.0, L_input,
            value=st.session_state[f'udl_start_var_{i}'],
            key=f'udl_start_sl_{i}',
            on_change=lambda idx=i: st.session_state.update({f'udl_start_var_{idx}': st.session_state[f'udl_start_sl_{idx}']})
        )
    with col2:
        st.number_input(
            label='Start',
            min_value=0.0,
            max_value=L_input,
            value=st.session_state[f'udl_start_var_{i}'],
            key=f'udl_start_in_{i}',
            format='%.3f',
            label_visibility='collapsed',
            on_change=lambda idx=i: st.session_state.update({f'udl_start_var_{idx}': st.session_state[f'udl_start_in_{idx}']})
        )
    init_var(f'udl_end_var_{i}', L_input)
    col3, col4 = st.sidebar.columns([3,1])
    with col3:
        st.slider(
            f'UDL {i+1} End ({length_unit})',
            0.0, L_input,
            value=st.session_state[f'udl_end_var_{i}'],
            key=f'udl_end_sl_{i}',
            on_change=lambda idx=i: st.session_state.update({f'udl_end_var_{idx}': st.session_state[f'udl_end_sl_{idx}']})
        )
    with col4:
        st.number_input(
            label='End',
            min_value=0.0,
            max_value=L_input,
            value=st.session_state[f'udl_end_var_{i}'],
            key=f'udl_end_in_{i}',
            format='%.3f',
            label_visibility='collapsed',
            on_change=lambda idx=i: st.session_state.update({f'udl_end_var_{idx}': st.session_state[f'udl_end_in_{idx}']})
        )
    st.sidebar.markdown(f"UDL {i+1} Intensity (N/{length_unit}):")
    init_var(f'udl_q_var_{i}', 100.0)
    col5, col6 = st.sidebar.columns([3,1])
    with col5:
        st.slider(
            f'UDL {i+1} Intensity (N/{length_unit})',
            0.0, 10000.0,
            value=st.session_state[f'udl_q_var_{i}'],
            key=f'udl_q_sl_{i}',
            on_change=lambda idx=i: st.session_state.update({f'udl_q_var_{idx}': st.session_state[f'udl_q_sl_{idx}']})
        )
    with col6:
        st.number_input(
            label='Intensity',
            min_value=0.0,
            max_value=10000.0,
            value=st.session_state[f'udl_q_var_{i}'],
            key=f'udl_q_in_{i}',
            format='%.1f',
            label_visibility='collapsed',
            on_change=lambda idx=i: st.session_state.update({f'udl_q_var_{idx}': st.session_state[f'udl_q_in_{idx}']})
        )
    s = st.session_state[f'udl_start_var_{i}']/factor
    e = st.session_state[f'udl_end_var_{i}']/factor
    q = st.session_state[f'udl_q_var_{i}']*factor
    udl_list.append((s, e, q))

# Plot layout
fig, ax = plt.subplots(figsize=(8,2))
ax.hlines(0, 0, L_input, linewidth=5, color='gray')
for p in support_positions:
    ax.scatter(p*factor, 0, marker='^', s=200, color='red')
for p, m in zip(load_positions, load_mags):
    ax.annotate('', xy=(p*factor, 0), xytext=(p*factor, 0.3), arrowprops=dict(arrowstyle='->', lw=1.5))
for s, e, _ in udl_list:
    ax.fill_between([s*factor, e*factor], 0.25, 0.35, alpha=0.3, color='blue')
ax.set_xlim(-L_input*0.05, L_input*1.05)
ax.set_ylim(-0.5, 0.6)
ax.set_yticks([])
ax.set_xlabel(f'Beam Length ({length_unit})')
ax.set_title('Beam: Supports ▲, Loads ↓, UDLs blue')
st.pyplot(fig)


# Prepare nodes
sup_nodes = [int(round(p/L*num_elem))+1 for p in support_positions]
pt_loads = [(int(round(p/L*num_elem))+1, m) for p, m in zip(load_positions, load_mags)]



# Display load summary in sidebar
st.sidebar.markdown("### Load Summary")
if n_loads > 0:
    st.sidebar.markdown("**Point Loads (N):**")
    for i, (pos, mag) in enumerate(zip(load_positions, load_mags)):
        st.sidebar.markdown(f"• Load {i+1}: {pos*factor:.2f} {length_unit}, {mag:.2f} N")

if n_udl > 0:
    st.sidebar.markdown(f"**UDLs (N/{length_unit}):**")
    for i in range(n_udl):
        start = st.session_state[f'udl_start_var_{i}']
        end = st.session_state[f'udl_end_var_{i}']
        intensity = st.session_state[f'udl_q_var_{i}']
        st.sidebar.markdown(f"• UDL {i+1}: {start:.2f}–{end:.2f} {length_unit}, {intensity:.2f} N/{length_unit}")



if st.button('Run Analysis'):
    coords, M, V, D = analyze_beam(L, sup_nodes, pt_loads, udl_list, E, I, num_elem)

    moment_factor = 1.0 if length_unit == 'm' else 1e3
    shear_factor = 1.0
    x=coords * factor
    # Bending Moment Diagram
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=coords * factor, y=M * moment_factor, mode='lines', name='Moment', line_color='#00008B',line_width=4.5))
    fig1.update_layout(xaxis_range=[min(x)-0.01*(max(x)-min(x)), max(x)+0.1*(max(x)-min(x))],
        title='Bending Moment Diagram',
        xaxis_title=f'Beam Length ({length_unit})',
        yaxis_title=f'Moment (N·{length_unit})',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', tickfont=dict(color='black'), title_font=dict(color='black'),tickformat=".1e"),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', tickfont=dict(color='black'), title_font=dict(color='black'),tickformat=".1e")
    ),
    #st.plotly_chart(fig1)

    # Shear Force Diagram
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=coords * factor, y=V * shear_factor, mode='lines', name='Shear', line_color='#00008B',line_width=4.5))
    fig2.update_layout(xaxis_range=[min(x)-0.01*(max(x)-min(x)), max(x)+0.1*(max(x)-min(x))],
        title='Shear Force Diagram',
        xaxis_title=f'Beam Length ({length_unit})',
        yaxis_title='Shear (N)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', tickfont=dict(color='black'), title_font=dict(color='black'),tickformat=".1e"),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', tickfont=dict(color='black'), title_font=dict(color='black'),tickformat=".1e")
    ),
    #st.plotly_chart(fig2)

    # Deflection Diagram
    node_coords = np.linspace(0, L, len(D))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=node_coords * factor, y=D * factor, mode='lines', name='Deflection', line_color='#00008B',line_width=4.5))
    fig3.update_layout(xaxis_range=[min(x)-0.01*(max(x)-min(x)), max(x)+0.1*(max(x)-min(x))],
        title='Deflection Diagram',
        xaxis_title=f'Beam Length ({length_unit})',
        yaxis_title=f'Deflection ({length_unit})',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(showgrid=True, gridcolor='black', linecolor='black', tickfont=dict(color='black'), title_font=dict(color='black'),tickformat=".1e"),
        yaxis=dict(showgrid=True, gridcolor='black', linecolor='black', tickfont=dict(color='black'), title_font=dict(color='black'),tickformat=".1e")
    ),
    #st.plotly_chart(fig3)


                                                 
                     
             
     
                                                                                       
                                                              