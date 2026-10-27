"""Start the Sermon-Snip interface:  python web_app.py   (the classic Tk window is still available as app.py)."""
import argparse
import threading
import time

import native
import server


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sermon-Snip web interface")
    ap.add_argument("--port", type=int, default=0, help="port to listen on (default: any free port)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a window automatically")
    args = ap.parse_args(argv)
    session = server.make_session()
    stop = threading.Event()
    app, httpd = server.start(session, args.port, on_quit=stop.set)
    url = f"http://127.0.0.1:{app.port}/"
    print(f"Sermon-Snip is running at {url}")
    print("Use the Quit button in the app (or press Ctrl+C here) to stop it.")
    if not args.no_browser:
        native.open_app_window(url)
    try:
        while not stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        httpd.shutdown()


if __name__ == "__main__":
    main()
