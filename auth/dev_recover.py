"""Operator-only synthetic password recovery;no public reset or email provider."""
import argparse
import getpass
from sqlalchemy import create_engine,text
from dev_portal import _require_synthetic,_password_hash,_now


def recover_dev_user(engine,tenant,username,password):
    _require_synthetic(engine)
    if not tenant.strip() or not username.strip() or max(len(tenant),len(username))>100:raise ValueError('Bounded tenant and username required.')
    if not 12<=len(password)<=128:raise ValueError('Password must be12to128characters.')
    # Hash first,then update credential,revoke sessions and audit atomically.
    encoded=_password_hash(password)
    with engine.begin() as connection:
        user=connection.execute(text('SELECT id,tenant,active FROM dev_users WHERE tenant=:t AND username=:u'),{'t':tenant.strip(),'u':username.strip()}).mappings().first()
        if not user or not user['active']:raise ValueError('Active synthetic account not found.No change made.')
        connection.execute(text('UPDATE dev_users SET password_hash=:p WHERE id=:u'),{'p':encoded,'u':user['id']})
        revoked=connection.execute(text('DELETE FROM dev_sessions WHERE user_id=:u'),{'u':user['id']}).rowcount
        # A local process is not the patient;never misattribute operator recovery.
        connection.execute(text("INSERT INTO dev_audit(tenant,actor_id,event,resource_type,resource_id,created_at) VALUES (:t,NULL,'local_operator_password_recovery','user',:u,:at)"),{'t':user['tenant'],'u':user['id'],'at':_now()})
        return {'sessions_revoked':revoked,'account_reactivated':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database-url',required=True);p.add_argument('--tenant',required=True);p.add_argument('--username',required=True);a=p.parse_args()
    first=getpass.getpass('New synthetic password (12–128characters): ');second=getpass.getpass('Confirm new password: ')
    if first!=second:p.exit(2,'Passwords did not match.No change made.\n')
    engine=create_engine(a.database_url)
    try:result=recover_dev_user(engine,a.tenant,a.username,first)
    except (RuntimeError,ValueError):p.exit(2,'Recovery refused.Verify the explicit synthetic flag,database,active account and password policy.\n')
    finally:engine.dispose()
    print('Synthetic credential replaced;all '+str(result['sessions_revoked'])+' existing account sessions revoked.No record erased or external message sent.')
if __name__=='__main__':main()
