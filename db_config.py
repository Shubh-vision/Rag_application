import psycopg2
import socket

def get_connection():
    try:
        print("Starting...")

        conn = psycopg2.connect(
            host="db.lgkygakecghsivtzgpmp.supabase.co",
            port=5432,
            database="postgres",
            user="postgres",
            password="Shubh12254##",
            sslmode = "require"
            )
        
        print("CONNECTING TO DB...")

        return conn   # 🔥 VERY IMPORTANT
        
    except Exception as e:
        print("❌ Error:", e)
        return None


get_connection()