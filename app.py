from datetime import date, datetime
from calendar import monthrange
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import ollama


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

CATEGORIES = [
    "Food",
    "Transport",
    "Education",
    "Entertainment",
    "Shopping",
    "Bills",
    "Health",
    "Travel",
    "Other",
]

PAYMENT_METHODS = [
    "Cash",
    "UPI",
    "Debit Card",
    "Credit Card",
    "Bank Transfer",
]

OLLAMA_MODEL = "llama3.2"


# ============================================================
# SESSION STATE / IN-MEMORY STORAGE
# ============================================================

def initialize_session_state():
    if "expenses" not in st.session_state:
        st.session_state.expenses = []

    if "budgets" not in st.session_state:
        st.session_state.budgets = {}

    if "category_budgets" not in st.session_state:
        st.session_state.category_budgets = {}

    if "next_expense_id" not in st.session_state:
        st.session_state.next_expense_id = 1


initialize_session_state()


# ============================================================
# HELPERS
# ============================================================

def get_month_key(selected_date: date) -> str:
    return selected_date.strftime("%Y-%m")


def get_month_range(selected_month: date):
    """
    Returns the first and last day of the selected month.

    Important:
    Both values are datetime.date objects.
    Do NOT call .date() on them.
    """
    start = selected_month.replace(day=1)

    last_day = monthrange(
        selected_month.year,
        selected_month.month,
    )[1]

    end = selected_month.replace(day=last_day)

    return start, end


def format_currency(value: float) -> str:
    return f"₹{value:,.2f}"


def expenses_to_dataframe(expenses=None):
    if expenses is None:
        expenses = st.session_state.expenses

    if not expenses:
        return pd.DataFrame(
            columns=[
                "id",
                "date",
                "category",
                "amount",
                "description",
                "payment_method",
                "created_at",
            ]
        )

    df = pd.DataFrame(expenses)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"])

    return df


def get_month_expenses(selected_month: date):
    start_date, end_date = get_month_range(selected_month)

    result = []

    for expense in st.session_state.expenses:
        expense_date = expense["date"]

        if isinstance(expense_date, datetime):
            expense_date = expense_date.date()

        if start_date <= expense_date <= end_date:
            result.append(expense)

    return result


def get_month_total(selected_month: date) -> float:
    expenses = get_month_expenses(selected_month)

    return sum(float(expense["amount"]) for expense in expenses)


def get_category_totals(selected_month: date):
    expenses = get_month_expenses(selected_month)

    totals = {}

    for expense in expenses:
        category = expense["category"]
        amount = float(expense["amount"])

        totals[category] = totals.get(category, 0) + amount

    return totals


def add_expense(
    expense_date: date,
    category: str,
    amount: float,
    description: str,
    payment_method: str,
):
    expense = {
        "id": st.session_state.next_expense_id,
        "date": expense_date,
        "category": category,
        "amount": float(amount),
        "description": description.strip(),
        "payment_method": payment_method,
        "created_at": datetime.now(),
    }

    st.session_state.expenses.append(expense)
    st.session_state.next_expense_id += 1


def delete_expense(expense_id: int):
    st.session_state.expenses = [
        expense
        for expense in st.session_state.expenses
        if expense["id"] != expense_id
    ]


# ============================================================
# AI
# ============================================================

def generate_ai_insights(
    selected_month: date,
    monthly_budget: float,
    category_budgets: dict,
):
    expenses = get_month_expenses(selected_month)

    if not expenses:
        return "There are no expenses for the selected month yet."

    total_spent = sum(float(expense["amount"]) for expense in expenses)

    category_totals = get_category_totals(selected_month)

    expense_data = []

    for expense in expenses:
        expense_data.append(
            {
                "date": str(expense["date"]),
                "category": expense["category"],
                "amount": expense["amount"],
                "description": expense["description"],
                "payment_method": expense["payment_method"],
            }
        )

    prompt = f"""
You are a helpful personal finance assistant for a college student.

Analyze the student's expenses and provide practical advice.

Selected month:
{selected_month.strftime("%B %Y")}

Monthly budget:
₹{monthly_budget:,.2f}

Total spent:
₹{total_spent:,.2f}

Remaining budget:
₹{monthly_budget - total_spent:,.2f}

Category budgets:
{category_budgets}

Category spending:
{category_totals}

Expense records:
{expense_data}

Provide:

1. A short spending summary.
2. The top spending categories.
3. Categories where spending is high compared with the budget.
4. Unusual or potentially unnecessary spending.
5. Three practical ways the student can save money.
6. A simple recommendation for the remaining days of the month.

Keep the answer concise and student-friendly.
Use Indian Rupee amounts.
Do not invent transactions or numbers.
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]

    except Exception as error:
        return (
            "Unable to generate AI insights.\n\n"
            f"Make sure Ollama is running and `{OLLAMA_MODEL}` is installed.\n\n"
            f"Error: {error}"
        )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    st.sidebar.title("💰 Smart Expense Tracker")

    st.sidebar.caption("Student Finance Manager")

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "➕ Add Expense",
            "📋 Expense History",
            "💰 Budget Planner",
            "📊 Analytics",
            "🤖 AI Insights",
        ],
    )

    st.sidebar.divider()

    st.sidebar.subheader("Quick Stats")

    total_expenses = len(st.session_state.expenses)

    total_amount = sum(
        float(expense["amount"])
        for expense in st.session_state.expenses
    )

    st.sidebar.metric(
        "Total Transactions",
        total_expenses,
    )

    st.sidebar.metric(
        "Total Spent",
        format_currency(total_amount),
    )

    st.sidebar.divider()

    if st.sidebar.button(
        "🗑️ Clear All Expenses",
        use_container_width=True,
    ):
        st.session_state.expenses = []
        st.session_state.next_expense_id = 1
        st.rerun()

    return page


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_page():
    st.title("🏠 Dashboard")

    st.caption("Track your spending and stay within your student budget.")

    selected_month = st.date_input(
        "Select month",
        value=date.today(),
    )

    selected_month = selected_month.replace(day=1)

    month_expenses = get_month_expenses(selected_month)

    total_spent = sum(
        float(expense["amount"])
        for expense in month_expenses
    )

    month_key = get_month_key(selected_month)

    monthly_budget = float(
        st.session_state.budgets.get(month_key, 0)
    )

    remaining = monthly_budget - total_spent

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Monthly Budget",
            format_currency(monthly_budget),
        )

    with col2:
        st.metric(
            "Spent",
            format_currency(total_spent),
        )

    with col3:
        st.metric(
            "Remaining",
            format_currency(remaining),
        )

    with col4:
        st.metric(
            "Transactions",
            len(month_expenses),
        )

    st.divider()

    if monthly_budget > 0:
        percentage = (total_spent / monthly_budget) * 100

        st.subheader("Budget Usage")

        st.progress(
            min(max(percentage / 100, 0), 1)
        )

        if percentage >= 100:
            st.error(
                f"You have exceeded your budget by "
                f"{format_currency(abs(remaining))}."
            )
        elif percentage >= 80:
            st.warning(
                f"You have used {percentage:.1f}% of your budget."
            )
        else:
            st.success(
                f"You have used {percentage:.1f}% of your budget."
            )
    else:
        st.info(
            "No monthly budget has been configured. "
            "Go to Budget Planner to set one."
        )

    st.divider()

    if month_expenses:
        st.subheader("Spending by Category")

        category_totals = get_category_totals(selected_month)

        category_df = pd.DataFrame(
            {
                "Category": list(category_totals.keys()),
                "Amount": list(category_totals.values()),
            }
        )

        fig = px.pie(
            category_df,
            names="Category",
            values="Amount",
            hole=0.45,
            title=f"Spending - {selected_month.strftime('%B %Y')}",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("Recent Expenses")

        df = expenses_to_dataframe(month_expenses)

        if not df.empty:
            df = df.sort_values(
                "date",
                ascending=False,
            )

            display_df = df[
                [
                    "date",
                    "category",
                    "amount",
                    "description",
                    "payment_method",
                ]
            ].head(5)

            display_df = display_df.rename(
                columns={
                    "date": "Date",
                    "category": "Category",
                    "amount": "Amount",
                    "description": "Description",
                    "payment_method": "Payment Method",
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info(
            "No expenses recorded for this month."
        )


# ============================================================
# ADD EXPENSE
# ============================================================

def add_expense_page():
    st.title("➕ Add Expense")

    st.caption("Add a new expense to your in-memory expense tracker.")

    with st.form("add_expense_form"):
        expense_date = st.date_input(
            "Date",
            value=date.today(),
        )

        category = st.selectbox(
            "Category",
            CATEGORIES,
        )

        amount = st.number_input(
            "Amount",
            min_value=0.0,
            step=10.0,
            format="%.2f",
        )

        description = st.text_input(
            "Description",
            placeholder="e.g. Lunch at college",
        )

        payment_method = st.selectbox(
            "Payment Method",
            PAYMENT_METHODS,
        )

        submitted = st.form_submit_button(
            "Add Expense",
            use_container_width=True,
        )

    if submitted:
        if amount <= 0:
            st.error("Please enter an amount greater than ₹0.")
            return

        if not description.strip():
            st.error("Please enter a description.")
            return

        add_expense(
            expense_date=expense_date,
            category=category,
            amount=amount,
            description=description,
            payment_method=payment_method,
        )

        st.success(
            f"Expense of {format_currency(amount)} added successfully."
        )

        st.rerun()


# ============================================================
# EXPENSE HISTORY
# ============================================================

def expense_history_page():
    st.title("📋 Expense History")

    st.caption("View, filter, and delete your expenses.")

    df = expenses_to_dataframe()

    if df.empty:
        st.info("No expenses available.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_categories = st.multiselect(
            "Category",
            CATEGORIES,
            default=[],
        )

    with col2:
        min_amount = st.number_input(
            "Minimum Amount",
            min_value=0.0,
            value=0.0,
            step=50.0,
        )

    with col3:
        max_amount = st.number_input(
            "Maximum Amount",
            min_value=0.0,
            value=0.0,
            step=50.0,
            help="Set to 0 to disable the maximum filter.",
        )

    filtered_df = df.copy()

    if selected_categories:
        filtered_df = filtered_df[
            filtered_df["category"].isin(
                selected_categories
            )
        ]

    filtered_df = filtered_df[
        filtered_df["amount"] >= min_amount
    ]

    if max_amount > 0:
        filtered_df = filtered_df[
            filtered_df["amount"] <= max_amount
        ]

    filtered_df = filtered_df.sort_values(
        "date",
        ascending=False,
    )

    st.write(
        f"Showing **{len(filtered_df)}** transaction(s)."
    )

    display_df = filtered_df[
        [
            "id",
            "date",
            "category",
            "amount",
            "description",
            "payment_method",
        ]
    ].rename(
        columns={
            "id": "ID",
            "date": "Date",
            "category": "Category",
            "amount": "Amount",
            "description": "Description",
            "payment_method": "Payment Method",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Delete Expense")

    expense_ids = filtered_df["id"].tolist()

    if expense_ids:
        selected_id = st.selectbox(
            "Select Expense ID",
            expense_ids,
        )

        if st.button(
            "Delete Selected Expense",
            type="secondary",
        ):
            delete_expense(int(selected_id))

            st.success(
                f"Expense #{selected_id} deleted."
            )

            st.rerun()


# ============================================================
# BUDGET PLANNER
# ============================================================

def budget_planner_page():
    st.title("💰 Budget Planner")

    st.caption(
        "Set a monthly budget and optional category budgets."
    )

    selected_month = st.date_input(
        "Select month",
        value=date.today(),
    )

    selected_month = selected_month.replace(day=1)

    month_key = get_month_key(selected_month)

    current_budget = float(
        st.session_state.budgets.get(
            month_key,
            0,
        )
    )

    st.subheader(
        f"Monthly Budget - {selected_month.strftime('%B %Y')}"
    )

    monthly_budget = st.number_input(
        "Monthly Budget",
        min_value=0.0,
        value=current_budget,
        step=500.0,
        format="%.2f",
    )

    if st.button(
        "Save Monthly Budget",
        use_container_width=True,
    ):
        st.session_state.budgets[
            month_key
        ] = monthly_budget

        st.success(
            f"Monthly budget saved: "
            f"{format_currency(monthly_budget)}"
        )

    st.divider()

    st.subheader("Category Budgets")

    current_category_budgets = (
        st.session_state.category_budgets
        .get(month_key, {})
    )

    category_values = {}

    columns = st.columns(2)

    for index, category in enumerate(CATEGORIES):
        with columns[index % 2]:
            current_value = float(
                current_category_budgets.get(
                    category,
                    0,
                )
            )

            category_values[category] = st.number_input(
                category,
                min_value=0.0,
                value=current_value,
                step=100.0,
                format="%.2f",
                key=f"budget_{month_key}_{category}",
            )

    if st.button(
        "Save Category Budgets",
        use_container_width=True,
    ):
        st.session_state.category_budgets[
            month_key
        ] = category_values

        st.success(
            "Category budgets saved successfully."
        )

    st.divider()

    month_expenses = get_month_expenses(
        selected_month
    )

    total_spent = sum(
        float(expense["amount"])
        for expense in month_expenses
    )

    remaining = monthly_budget - total_spent

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Budget",
            format_currency(monthly_budget),
        )

    with col2:
        st.metric(
            "Spent",
            format_currency(total_spent),
        )

    with col3:
        st.metric(
            "Remaining",
            format_currency(remaining),
        )


# ============================================================
# ANALYTICS
# ============================================================

def analytics_page():
    st.title("📊 Analytics")

    st.caption(
        "Understand your spending patterns."
    )

    df = expenses_to_dataframe()

    if df.empty:
        st.info(
            "Add some expenses to see analytics."
        )
        return

    df["month"] = pd.to_datetime(
        df["date"]
    ).dt.to_period("M").astype(str)

    monthly_df = (
        df.groupby("month", as_index=False)["amount"]
        .sum()
    )

    st.subheader("Monthly Spending")

    fig = px.bar(
        monthly_df,
        x="month",
        y="amount",
        title="Monthly Spending",
        labels={
            "month": "Month",
            "amount": "Amount (₹)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Category Spending")

    category_df = (
        df.groupby(
            "category",
            as_index=False,
        )["amount"]
        .sum()
        .sort_values(
            "amount",
            ascending=False,
        )
    )

    fig = px.bar(
        category_df,
        x="category",
        y="amount",
        title="Spending by Category",
        labels={
            "category": "Category",
            "amount": "Amount (₹)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Payment Method")

    payment_df = (
        df.groupby(
            "payment_method",
            as_index=False,
        )["amount"]
        .sum()
    )

    fig = px.pie(
        payment_df,
        names="payment_method",
        values="amount",
        hole=0.45,
        title="Spending by Payment Method",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Daily Spending")

    daily_df = (
        df.groupby(
            "date",
            as_index=False,
        )["amount"]
        .sum()
        .sort_values("date")
    )

    fig = px.line(
        daily_df,
        x="date",
        y="amount",
        markers=True,
        title="Daily Spending",
        labels={
            "date": "Date",
            "amount": "Amount (₹)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# AI INSIGHTS PAGE
# ============================================================

def ai_insights_page():
    st.title("🤖 AI Financial Insights")

    st.caption(
        f"Powered by local Ollama `{OLLAMA_MODEL}`"
    )

    selected_month = st.date_input(
        "Select month",
        value=date.today(),
    )

    selected_month = selected_month.replace(day=1)

    month_key = get_month_key(selected_month)

    monthly_budget = float(
        st.session_state.budgets.get(
            month_key,
            0,
        )
    )

    category_budgets = (
        st.session_state.category_budgets.get(
            month_key,
            {},
        )
    )

    month_expenses = get_month_expenses(
        selected_month
    )

    total_spent = sum(
        float(expense["amount"])
        for expense in month_expenses
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Monthly Budget",
            format_currency(monthly_budget),
        )

    with col2:
        st.metric(
            "Spent",
            format_currency(total_spent),
        )

    with col3:
        st.metric(
            "Remaining",
            format_currency(
                monthly_budget - total_spent
            ),
        )

    st.divider()

    if not month_expenses:
        st.info(
            "Add expenses for this month before generating AI insights."
        )
        return

    st.subheader("AI Analysis")

    st.write(
        "The local Llama model will analyze your spending "
        "and provide personalized recommendations."
    )

    if st.button(
        "🤖 Generate AI Insights",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "Llama 3.2 is analyzing your expenses..."
        ):
            insights = generate_ai_insights(
                selected_month=selected_month,
                monthly_budget=monthly_budget,
                category_budgets=category_budgets,
            )

        st.markdown("### 💡 Recommendations")

        st.markdown(insights)

    st.divider()

    st.subheader("Current Month Spending")

    category_totals = get_category_totals(
        selected_month
    )

    if category_totals:
        category_df = pd.DataFrame(
            {
                "Category": list(
                    category_totals.keys()
                ),
                "Amount": list(
                    category_totals.values()
                ),
            }
        )

        fig = px.pie(
            category_df,
            names="Category",
            values="Amount",
            hole=0.45,
            title="Category Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# MAIN APP
# ============================================================

def main():
    page = render_sidebar()

    if page == "🏠 Dashboard":
        dashboard_page()

    elif page == "➕ Add Expense":
        add_expense_page()

    elif page == "📋 Expense History":
        expense_history_page()

    elif page == "💰 Budget Planner":
        budget_planner_page()

    elif page == "📊 Analytics":
        analytics_page()

    elif page == "🤖 AI Insights":
        ai_insights_page()


if __name__ == "__main__":
    main()
