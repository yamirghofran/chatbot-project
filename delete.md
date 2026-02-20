 How to do this again in the future:                                                                                                                      
                                                                                                                                                           
  # 1. Wipe everything (kills stuck connections + drops all tables)                                                                                        
  .venv/bin/python -c "                                                                                                                                    
  from bookdb.db.session import engine
  from sqlalchemy import text                                                                                                                              
  with engine.connect() as conn:                                                                                                                           
      conn.execute(text(\"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database() AND pid != pg_backend_pid()\"))        
      conn.commit()                                                                                                                                        
      conn.execute(text('DROP SCHEMA public CASCADE'))                                                                                                     
      conn.execute(text('CREATE SCHEMA public'))                                                                                                           
      conn.commit()                                                                                                                                        
  print('Database wiped.')                                                                                                                                 
  "                                                                                                                                                        
                                                                                                                                                           
  # 2. Rebuild all tables from migrations                                                                                                                  
  .venv/bin/alembic upgrade head                                                                                                                           
                                                                                                                                                           
  The key step is pg_terminate_backend — that kills any stuck connections (like from a hanging marimo notebook) that would otherwise hold locks and block  
  the DROP.                                                                                                                                                
               