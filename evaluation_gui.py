import streamlit as st
import json
import pandas as pd
import random

# --- Configuration ---
MANUAL_FILE = 'key points/manual_answer_key_facts.json'
MODEL_FILE = 'key points/model_answer_key_facts_gemini_rag.json'
OUTPUT_FILE = 'evaluation_results.json'
SAMPLE_FRACTION = 0.1

# --- Helper Functions ---

def load_and_prepare_data():
    """Loads, merges, and samples the data for evaluation."""
    try:
        # Load the JSON files into pandas DataFrames
        manual_df = pd.read_json(MANUAL_FILE)
        model_df = pd.read_json(MODEL_FILE)

        # Merge the two dataframes on 'id' and 'question'
        # Suffixes are added to differentiate the columns from each file
        merged_df = pd.merge(
            manual_df,
            model_df,
            on=['id', 'question'],
            suffixes=('_manual', '_model')
        )

        # Randomly sample a fraction of the questions
        sampled_df = merged_df.sample(frac=SAMPLE_FRACTION, random_state=42)
        
        # Convert DataFrame to a list of dictionaries for easier handling
        return sampled_df.to_dict('records')
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please make sure the JSON files are in the 'key points' directory.")
        return None

def render_ground_truth_facts(key_facts, color, q_id=None, safety_critical_list=None):
    """Renders a list of ground truth key facts with checkboxes for recall evaluation."""
    if not key_facts:
        return []

    selections = []
    for i, fact in enumerate(key_facts):
        cols = st.columns([1, 20])
        with cols[0]:
            key = f"gt_{q_id}_{color}_{i}"
            is_checked = st.checkbox("", key=key, value=st.session_state.get(key, False))
            selections.append(is_checked)
        with cols[1]:
            fact_html = f'<p style="color:black;">{i+1}. {fact}</p>'
            st.markdown(fact_html, unsafe_allow_html=True)
            
    return selections

def render_display_only_facts(key_facts, color):
    """Renders a list of key facts as styled text without any interactive elements."""
    if not key_facts:
        return

    for i, fact in enumerate(key_facts):
        fact_html = f'<p style="color:black;">{i+1}. {fact}</p>'
        st.markdown(fact_html, unsafe_allow_html=True)

def render_model_facts_evaluation(key_facts, color, q_id, type_key):
    """
    Renders model key facts with a custom-built, perfectly aligned radio
    button system using st.columns and st.checkbox.
    """
    if not key_facts:
        st.write("No key facts provided.")
        return [], [], []

    # --- Header ---
    # Create a header with the same column structure as the checkboxes below for perfect alignment.
    header_cols = st.columns([10, 10])
    with header_cols[1]:
        label_cols = st.columns(3)
        with label_cols[0]:
            st.markdown("<p style='text-align: center;'><b>Accurate</b></p>", unsafe_allow_html=True)
        with label_cols[1]:
            st.markdown("<p style='text-align: center;'><b>Inaccurate</b></p>", unsafe_allow_html=True)
        with label_cols[2]:
            st.markdown("<p style='text-align: center;'><b>Irrelevant</b></p>", unsafe_allow_html=True)

    all_selections = []

    for i, fact in enumerate(key_facts):
        # --- State Management ---
        # A single key in session_state stores the choice ('Accurate', 'Inaccurate', 'Irrelevant').
        eval_key = f"eval_{q_id}_{type_key}_{i}"
        # Default to 'Irrelevant' if not set
        if eval_key not in st.session_state:
            st.session_state[eval_key] = 'Irrelevant'

        # --- Widget Rendering ---
        row_cols = st.columns([10, 10])

        with row_cols[0]: # Column for the fact text
            fact_html = f'<p style="color:black;">{i+1}. {fact}</p>'
            st.markdown(fact_html, unsafe_allow_html=True)

        with row_cols[1]: # Columns for the custom "radio-checkboxes"
            selection_cols = st.columns(3)
            with selection_cols[0]:
                # Center the checkbox using an empty column
                l,c,r = st.columns([1,1,1])
                with c:
                    is_accurate = st.checkbox("Accurate", value=(st.session_state[eval_key] == 'Accurate'), key=f"acc_{eval_key}", label_visibility="collapsed")
            with selection_cols[1]:
                l,c,r = st.columns([1,1,1])
                with c:
                    is_inaccurate = st.checkbox("Inaccurate", value=(st.session_state[eval_key] == 'Inaccurate'), key=f"inacc_{eval_key}", label_visibility="collapsed")
            with selection_cols[2]:
                l,c,r = st.columns([1,1,1])
                with c:
                    is_irrelevant = st.checkbox("Irrelevant", value=(st.session_state[eval_key] == 'Irrelevant'), key=f"irr_{eval_key}", label_visibility="collapsed")

        # --- Logic to enforce radio button behavior ---
        current_state = st.session_state[eval_key]
        
        # Determine which checkbox was just clicked by comparing its state to the central state
        if is_accurate and current_state != 'Accurate':
            st.session_state[eval_key] = 'Accurate'
            st.rerun()
        elif is_inaccurate and current_state != 'Inaccurate':
            st.session_state[eval_key] = 'Inaccurate'
            st.rerun()
        elif is_irrelevant and current_state != 'Irrelevant':
            st.session_state[eval_key] = 'Irrelevant'
            st.rerun()
        # Handle case where user unchecks the active box.
        # This will make them all False. We default back to Irrelevant.
        elif not any([is_accurate, is_inaccurate, is_irrelevant]):
            st.session_state[eval_key] = 'Irrelevant'
            st.rerun()
            
        all_selections.append(st.session_state[eval_key])

    # Convert text selections into metric lists for saving
    accurate = [1 if s == "Accurate" else 0 for s in all_selections]
    inaccurate = [1 if s == "Inaccurate" else 0 for s in all_selections]
    irrelevant = [1 if s == "Irrelevant" else 0 for s in all_selections]

    return accurate, inaccurate, irrelevant

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="QA Evaluation Assistant")

# Custom CSS to improve font clarity and contrast
st.markdown("""
<style>
    .stApp {
        background-color: white;
    }
    /* Set a readable font and default black color for all text */
    body, .stApp, .stMarkdown, h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        color: black;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    h1 { font-weight: 700; }
    h2, h3 { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("Question Answering Evaluation")

with st.expander("Show Evaluation Instructions", expanded=True):
    st.markdown("""
    ### Welcome to the Answer Evaluation Tool!

    This tool breaks down evaluation into three steps: **Completeness**, **Correctness**, and **Relevance**.

    **Step 1: Evaluate Completeness and Risk Awareness**
    - Look at the **Ground Truth Key Points** on the left.
    - For each point, **check the box** if it's covered in the AI's full answer on the right.

    **Step 2: Evaluate Correctness (Precision)**
    - Look at the **AI Assistant's Key Points** on the right. For each point:
    - Mark it as **"Accurate"** if it is factually correct (according to the ground truth).
    - Mark it as **"Inaccurate"** if it is factually incorrect.
    - Mark it as **"Not Mentioned"** if this point is not found in the ground truth answer.

    **Step 3: Evaluate Relevance**
    - Look at the **AI Assistant's Key Points**.
    - For each point, **uncheck the box** if it is NOT relevant to the question that was asked. By default, all points are considered relevant.

    **Step 4: Actionability Assessment**
    - Look at the **AI Assistant's Key Points**.
    - For each point, **uncheck the box** if it is NOT relevant to the question that was asked. By default, all points are considered relevant.

    **Actionability Definition:**
    An answer is actionable if it enables the user to take a concrete next step, complete a task, or meaningfully advance their understanding. Actionable answers may include instructions, examples, recommendations, or clearly usable insights.

    **Color Guide:**
    - <p style='color:green; display:inline-block; font-weight:bold;'>Green:</p> <span style='font-weight:bold;'>Essential</span> facts.
    - <p style='color:royalblue; display:inline-block; font-weight:bold;'>Blue:</p> <span style='font-weight:bold;'>Optional</span> facts.
    - <p style='color:red; display:inline-block; font-weight:bold;'>⚠️:</p> <span style='font-weight:bold;'>Safety Critical</span> facts.
    """, unsafe_allow_html=True)

st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:gray;margin-top:2rem;margin-bottom:2rem">', unsafe_allow_html=True)

# --- Initialization ---
if 'questions' not in st.session_state:
    st.session_state.questions = load_and_prepare_data()
    st.session_state.current_index = 0
    st.session_state.results = {}

# --- Main Logic ---
if st.session_state.questions:
    
    total_questions = len(st.session_state.questions)
    current_q_data = st.session_state.questions[st.session_state.current_index]
    q_id = current_q_data['id']

    st.header(f"Question {st.session_state.current_index + 1}/{total_questions}")
    st.markdown(f"**Question:** {current_q_data['question']}")
    
    # --- Part 1: Completeness (Recall) ---
    st.subheader("Step 1: Evaluate Completeness and Risk Awareness")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander("Show Full Ground Truth Answer"):
            st.write(current_q_data['manual_answer'])
    with exp_col2:
        with st.expander("Show Full Assistant Answer"):
            st.write(current_q_data['model_answer'])
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("#### Ground Truth Key Points")
        st.markdown("_Check the box for each point covered by the AI._")
        gt_essential = current_q_data['key_facts_manual']['essential']
        gt_optional = current_q_data['key_facts_manual']['optional']
        gt_safety_critical = current_q_data['key_facts_manual']['safety_critical']
        gt_essential_selections = render_ground_truth_facts(gt_essential, 'green', q_id=q_id, safety_critical_list=gt_safety_critical)
        gt_optional_selections = render_ground_truth_facts(gt_optional, 'royalblue', q_id=q_id, safety_critical_list=gt_safety_critical)
        
    with comp_col2:
        st.markdown("#### AI Assistant's Key Points (for reference)")
        st.markdown("_&nbsp;_") # To align with subheader on left
        model_essential = current_q_data['key_facts_model']['essential']
        model_optional = current_q_data['key_facts_model']['optional']
        render_display_only_facts(model_essential, 'green')
        render_display_only_facts(model_optional, 'royalblue')

    st.markdown('<hr style="height:1px;border-width:0;color:gray;background-color:gray;margin-top:10rem;margin-bottom:2rem">', unsafe_allow_html=True)

    # --- Part 2: Correctness (Precision) ---
    st.subheader("Step 2: Evaluate Correctness")
    st.markdown(f"**Question:** {current_q_data['question']}")

    corr_col1, corr_col2 = st.columns(2)

    with corr_col1:
        st.markdown("#### Ground Truth Key Points (for reference)")
        st.markdown("_&nbsp;_") # To align
        gt_essential = current_q_data['key_facts_manual']['essential']
        gt_optional = current_q_data['key_facts_manual']['optional']
        render_display_only_facts(gt_essential, 'green')
        render_display_only_facts(gt_optional, 'royalblue')

    with corr_col2:
        st.markdown("#### AI Assistant's Key Points")
        st.markdown("_Select whether each point is Accurate, Inaccurate, or Not Mentioned._")
        
        def render_accuracy_evaluation(key_facts, color, q_id, type_key):
            if not key_facts:
                st.write("No key points to evaluate for accuracy.")
                return [], [], []
            
            # --- Header ---
            header_cols = st.columns([10, 10])
            with header_cols[1]:
                label_cols = st.columns(3)
                with label_cols[0]:
                    st.markdown("<p style='text-align: center;'><b>Accurate</b></p>", unsafe_allow_html=True)
                with label_cols[1]:
                    st.markdown("<p style='text-align: center;'><b>Inaccurate</b></p>", unsafe_allow_html=True)
                with label_cols[2]:
                    st.markdown("<p style='text-align: center;'><b>Not Mentioned</b></p>", unsafe_allow_html=True)

            all_selections = []

            for i, fact in enumerate(key_facts):
                eval_key = f"eval_{q_id}_{type_key}_{i}"
                if eval_key not in st.session_state or st.session_state[eval_key] not in ['Accurate', 'Inaccurate', 'Not Mentioned']:
                    st.session_state[eval_key] = 'Not Mentioned'

                row_cols = st.columns([10, 10])
                with row_cols[0]:
                    fact_html = f'<p style="color:black;">{i+1}. {fact}</p>'
                    st.markdown(fact_html, unsafe_allow_html=True)
                
                with row_cols[1]:
                    selection_cols = st.columns(3)
                    with selection_cols[0]:
                        l,c,r = st.columns([1,1,1])
                        with c:
                            is_accurate = st.checkbox("Accurate", value=(st.session_state[eval_key] == 'Accurate'), key=f"acc_{eval_key}", label_visibility="collapsed")
                    with selection_cols[1]:
                        l,c,r = st.columns([1,1,1])
                        with c:
                            is_inaccurate = st.checkbox("Inaccurate", value=(st.session_state[eval_key] == 'Inaccurate'), key=f"inacc_{eval_key}", label_visibility="collapsed")
                    with selection_cols[2]:
                        l,c,r = st.columns([1,1,1])
                        with c:
                            is_not_mentioned = st.checkbox("Not Mentioned", value=(st.session_state[eval_key] == 'Not Mentioned'), key=f"not_men_{eval_key}", label_visibility="collapsed")

                current_state = st.session_state[eval_key]
                if is_accurate and current_state != 'Accurate':
                    st.session_state[eval_key] = 'Accurate'
                    st.rerun()
                elif is_inaccurate and current_state != 'Inaccurate':
                    st.session_state[eval_key] = 'Inaccurate'
                    st.rerun()
                elif is_not_mentioned and current_state != 'Not Mentioned':
                    st.session_state[eval_key] = 'Not Mentioned'
                    st.rerun()
                elif not any([is_accurate, is_inaccurate, is_not_mentioned]):
                    st.session_state[eval_key] = 'Not Mentioned'
                    st.rerun()
                
                all_selections.append(st.session_state[eval_key])

            accurate = [1 if s == "Accurate" else 0 for s in all_selections]
            inaccurate = [1 if s == "Inaccurate" else 0 for s in all_selections]
            not_mentioned = [1 if s == "Not Mentioned" else 0 for s in all_selections]

            return accurate, inaccurate, not_mentioned

        # Essential and Optional key points evaluation
        acc_ess, inacc_ess, not_men_ess = render_accuracy_evaluation(model_essential, 'green', q_id, 'essential')
        acc_opt, inacc_opt, not_men_opt = render_accuracy_evaluation(model_optional, 'royalblue', q_id, 'optional')

    st.markdown('<hr style="height:0.5px;border-width:0;color:gray;background-color:gray;margin-top:3rem;margin-bottom:1rem">', unsafe_allow_html=True)

    # --- Part 3: Relevance Evaluation ---
    st.subheader("Step 3: Evaluate Relevance")

    def render_relevance_evaluation(key_facts, color, q_id, type_key):
        """Renders model key facts with checkboxes for relevance evaluation."""
        if not key_facts:
            st.write("No key facts provided.")
            return []

        selections = []
        for i, fact in enumerate(key_facts):
            relevance_key = f"relevance_{q_id}_{type_key}_{i}"
            
            row_cols = st.columns([1, 20])
            with row_cols[0]:
                is_relevant = st.checkbox("", key=relevance_key, value=st.session_state.get(relevance_key, True))
                selections.append(is_relevant)
            with row_cols[1]:
                fact_html = f'<p style="color:black;">{i+1}. {fact}</p>'
                st.markdown(fact_html, unsafe_allow_html=True)
                
        return selections

    rel_col1, rel_col2 = st.columns(2)
    with rel_col1:
        st.write("**Question:**", current_q_data['question'])

    with rel_col2:
        st.markdown("#### AI Assistant's Key Points")
        st.markdown("_Uncheck the box for any point that does not address the question._")
        model_essential = current_q_data['key_facts_model']['essential']
        rel_ess_selections = render_relevance_evaluation(model_essential, 'green', q_id, 'essential')
        rel_opt_selections = render_relevance_evaluation(model_optional, 'royalblue', q_id, 'optional')

    st.markdown('<hr style="height:0.5px;border-width:0;color:gray;background-color:gray;margin-top:3rem;margin-bottom:1rem">', unsafe_allow_html=True)

    # --- Part 4: Actionability Assessment ---
    st.subheader("Step 4: Actionability Assessment")
    
    act_col1, act_col2 = st.columns(2)
    
    with act_col1:
        st.markdown("**Instructions:** Click the checkbox if the AI Assistant's key points represent an actionable response.")
        
        actionability_key = f"actionability_{q_id}"
        is_actionable = st.checkbox(
            "The assistant response represents actionability (is actionable)",
            key=actionability_key,
            value=st.session_state.get(actionability_key, False)
        )
    
    with act_col2:
        st.markdown("#### AI Assistant's Key Points (for reference)")
        model_essential = current_q_data['key_facts_model']['essential']
        model_optional = current_q_data['key_facts_model']['optional']
        render_display_only_facts(model_essential, 'green')
        render_display_only_facts(model_optional, 'royalblue')

    def save_current_state():
        # Calculate safety-critical coverage automatically
        def check_safety_critical_coverage():
            if not gt_safety_critical:
                return 0
            
            covered_count = 0
            ai_answer_lower = current_q_data['model_answer'].lower()
            
            for safety_point in gt_safety_critical:
                # Simple keyword matching - check if key words from safety point appear in AI answer
                safety_words = safety_point.lower().split()
                # Check if at least 70% of the key words from safety point are in AI answer
                matching_words = sum(1 for word in safety_words if len(word) > 3 and word in ai_answer_lower)
                if matching_words >= len(safety_words) * 0.7:
                    covered_count += 1
            
            return covered_count
        
        safety_critical_covered = check_safety_critical_coverage()
        
        # Completeness metrics
        st.session_state.results[q_id] = {
            'question_id': q_id,
            'question_text': current_q_data['question'],
            'total_essential_ground_truth': len(gt_essential),
            'covered_essential_ground_truth': sum(gt_essential_selections),
            'total_optional_ground_truth': len(gt_optional),
            'covered_optional_ground_truth': sum(gt_optional_selections),
            'total_safety_critical_ground_truth': len(gt_safety_critical),
            'covered_safety_critical_ground_truth': safety_critical_covered,
            # Correctness metrics
            'total_essential_model': len(model_essential),
            'accurate_essential_model': sum(acc_ess),
            'inaccurate_essential_model': sum(inacc_ess),
            'not_mentioned_essential_model': sum(not_men_ess),
            'total_optional_model': len(model_optional),
            'accurate_optional_model': sum(acc_opt),
            'inaccurate_optional_model': sum(inacc_opt),
            'not_mentioned_optional_model': sum(not_men_opt),
            # Relevance metrics
            'relevant_essential_model': sum(rel_ess_selections),
            'irrelevant_essential_model': len(model_essential) - sum(rel_ess_selections),
            'relevant_optional_model': sum(rel_opt_selections),
            'irrelevant_optional_model': len(model_optional) - sum(rel_opt_selections),
            # Actionability assessment
            'is_actionable': is_actionable,
        }

    def save_results_to_file(show_success=False):
        """Saves the collected results from session state to the output JSON file."""
        if 'results' not in st.session_state or not st.session_state.results:
            return # Don't save if there are no results
            
        final_results = list(st.session_state.results.values())
        try:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(final_results, f, indent=4)
            if show_success:
                st.sidebar.success(f"Results successfully saved to {OUTPUT_FILE}")
                st.balloons()
        except Exception as e:
            st.sidebar.error(f"Error saving file: {e}")

    st.sidebar.title("Navigation")
    
    nav_cols = st.sidebar.columns(2)
    with nav_cols[0]:
        if st.button("⬅️ Previous", disabled=(st.session_state.current_index == 0)):
            save_current_state()
            save_results_to_file()
            st.session_state.current_index -= 1
            st.rerun()

    with nav_cols[1]:
        if st.button("Next ➡️", disabled=(st.session_state.current_index >= total_questions - 1)):
            save_current_state()
            save_results_to_file()
            st.session_state.current_index += 1
            st.rerun()
            
    st.sidebar.progress((st.session_state.current_index + 1) / total_questions)

    if st.sidebar.button("Finish and Save Results"):
        save_current_state()
        save_results_to_file(show_success=True)

else:
    st.warning("No questions loaded. Please check the file paths and try again.") 