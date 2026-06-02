import psycopg2

# Define the database connection string
LUCIDOTA_GO_STATE_DSN = 'your_database_connection_string'

def river_status_dashboard():
    # Establish a connection to the database
    conn = psycopg2.connect(LUCIDOTA_GO_STATE_DSN)
    cur = conn.cursor()

    # Query 1: Overall training row statistics
    cur.execute("""
        SELECT COUNT(*), AVG(actual_gain), MIN(created_at), MAX(created_at) 
        FROM lucidota_control.river_training_row;
    """)
    overall_stats = cur.fetchone()

    # Query 2: Model usage and performance statistics
    cur.execute("""
        SELECT model_used, COUNT(*), AVG(actual_gain) 
        FROM lucidota_control.river_training_row 
        GROUP BY model_used 
        ORDER BY 3 DESC;
    """)
    model_stats = cur.fetchall()

    # Query 3: Non-default river score count
    cur.execute("""
        SELECT COUNT(*) 
        FROM lucidota_learning.river_score 
        WHERE river_prediction != 0.5;
    """)
    non_default_score_count = cur.fetchone()[0]

    # Query 4: Unconsumed operator feedback signal count
    cur.execute("""
        SELECT COUNT(*) 
        FROM lucidota_learning.operator_feedback_signal 
        WHERE consumed_at IS NULL;
    """)
    unconsumed_feedback_count = cur.fetchone()[0]

    # Print the dashboard
    print("River Status Dashboard")
    print("------------------------")
    print(f"Overall Training Rows: {overall_stats[0]}")
    print(f"Average Actual Gain: {overall_stats[1]}")
    print(f"Oldest Training Row: {overall_stats[2]}")
    print(f"Newest Training Row: {overall_stats[3]}")
    print()
    print("Model Usage and Performance:")
    for model, count, avg_gain in model_stats:
        print(f"Model: {model}, Count: {count}, Average Gain: {avg_gain}")
    print()
    print(f"Non-Default River Score Count: {non_default_score_count}")
    print(f"Unconsumed Operator Feedback Signal Count: {unconsumed_feedback_count}")

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    river_status_dashboard()

Replace `'your_database_connection_string'` with your actual database connection string. This script will print a dashboard with the results of the four queries, providing a quick status check for River's learning progress.
