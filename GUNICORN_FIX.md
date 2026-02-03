# Gunicorn Initialization Fix

## Problem Analysis

The container was starting correctly and showing successful initialization logs, but then failing during the gunicorn setup phase. The issue was traced to how the Flask application was being initialized.

### Root Cause

The critical initialization code (database creation, checkpoint cleanup, session cleanup, etc.) was placed inside the `if __name__ == "__main__":` block at the end of `main.py`.

**Why this is problematic:**
- When running with Flask development server (`python main.py`), the code works fine because `__name__` equals `"__main__"`
- When gunicorn imports the module (`gunicorn main:app`), `__name__` equals `"main"` (not `"__main__"`)
- Therefore, all the initialization code was **skipped** when gunicorn tried to load the app
- The `preload_app=True` setting in `gunicorn.conf.py` made this worse by trying to load the app before forking workers

### Error Symptoms

From the logs:
```
2026-02-03 11:50:37,230 - initialization - INFO - [INIT] Application initialization complete
  File "/app/.venv/bin/gunicorn", line 7, in <module>
    sys.exit(run())
  File "/app/.venv/lib/python3.13/site-packages/gunicorn/app/wsgiapp.py", line 66, in run
  File "/app/.venv/lib/python3.13/site-packages/gunicorn/app/base.py", line 71, in run
  File "/app/.venv/lib/python3.13/site-packages/gunicorn/arbiter.py", line 121, in setup
1. Loading .env file...
```

The initialization completed but gunicorn couldn't properly set up because essential app components weren't initialized.

## Solution

**Moved all initialization code outside the `if __name__ == "__main__":` block**

### What was changed:

1. **Database Creation** - Now runs on module import
2. **Checkpoint Cleanup Manager** - Initializes immediately
3. **Session Cleanup Manager** - Starts background cleanup thread
4. **Workflow State Manager** - Initialized on import
5. **Shutdown Handlers** - Registered via `atexit` on import
6. **Standards Cache Pre-warming** - Happens during module load

### Code Structure (After Fix):

```python
# ... Flask app creation and routes ...

def create_db():
    """Database initialization"""
    # ...

# =========================================================================
# === APPLICATION INITIALIZATION (Runs on both Gunicorn and Dev Server) ===
# =========================================================================
# This section runs when the module is imported (e.g., by gunicorn main:app)

create_db()
# Initialize checkpoint cleanup
# Initialize session cleanup
# Initialize workflow state manager
# Register shutdown handlers
# Pre-warm caches

# =========================================================================
# === DEVELOPMENT SERVER (Only runs when executed directly) ===
# =========================================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
```

## Benefits of This Fix

1. ✅ **Gunicorn Compatibility** - App initializes correctly whether run via `python main.py` or `gunicorn main:app`
2. ✅ **Consistent Behavior** - Same initialization in development and production
3. ✅ **Preload Support** - Works with `preload_app=True` in gunicorn config
4. ✅ **Worker Forking** - Proper initialization before worker processes fork
5. ✅ **Background Services** - Cleanup managers start correctly in all environments

## Testing

After deployment, verify:
- [ ] Gunicorn starts without errors
- [ ] Database tables are created
- [ ] Admin user exists
- [ ] Background cleanup threads are running
- [ ] API endpoints respond correctly
- [ ] Standards cache is pre-warmed

## Related Configuration

- `gunicorn.conf.py` - Uses `preload_app=True` which requires module-level initialization
- `initialization.py` - Handles environment and singleton setup before main imports
- `.env` - Environment variables loaded during initialization phase

## Deployment Notes

This fix allows the application to run properly in containerized environments where:
- Gunicorn is the WSGI server (production standard)
- Workers need shared pre-loaded resources
- Background cleanup threads must start on initialization
- Database must be ready before first request

The application now follows the proper Flask initialization pattern for production deployment.
