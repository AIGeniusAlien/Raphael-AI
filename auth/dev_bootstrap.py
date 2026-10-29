"""Provision synthetic portal accounts without defaults or command-line passwords."""

import argparse
import getpass
import sys

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from dev_portal import create_dev_user, init_portal_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True, help="Isolated synthetic SQLite URL, never production")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", required=True, choices=("patient", "clinician", "org_admin"))
    parser.add_argument("--password-stdin", action="store_true", help="Read one password line from stdin; otherwise prompt privately")
    args = parser.parse_args()
    password = sys.stdin.readline().rstrip("\r\n") if args.password_stdin else getpass.getpass("Synthetic development password (12–128 characters): ")
    engine = create_engine(args.database_url)
    try:
        init_portal_schema(engine)
        user = create_dev_user(engine, args.tenant, args.username, password, args.role)
    except IntegrityError:
        parser.exit(2, "Account already exists; credentials and data were not overwritten.\n")
    except (RuntimeError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    finally:
        engine.dispose()
    print(f"Created synthetic {user['role']} account {user['username']} in tenant {user['tenant']}.")


if __name__ == "__main__":
    main()
