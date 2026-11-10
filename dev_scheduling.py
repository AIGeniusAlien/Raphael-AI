"""Synthetic-only appointment scheduling with atomic resource conflict checks.

This is a development workflow, not a live clinical booking or notification service.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from database import execute_sqlite_script
from dev_portal import _require_synthetic
from scheduling_core import validate_interval

SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_appointments (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 tenant TEXT NOT NULL, patient_id INTEGER NOT NULL REFERENCES dev_patients(id),
 created_by INTEGER NOT NULL REFERENCES dev_users(id),
 start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('scheduled','cancelled')),
 reason TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_appointment_resources (
 appointment_id INTEGER NOT NULL REFERENCES dev_appointments(id),
 resource TEXT NOT NULL, PRIMARY KEY(appointment_id,resource)
);
CREATE INDEX IF NOT EXISTS dev_appointments_tenant_window
 ON dev_appointments(tenant,status,start_utc,end_utc);
CREATE INDEX IF NOT EXISTS dev_appointment_resources_lookup
 ON dev_appointment_resources(resource,appointment_id);
"""

def init_scheduling_schema(engine):
    _require_synthetic(engine)
    with engine.begin() as c:
        execute_sqlite_script(c, SCHEMA)

class AppointmentInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    patient_id:int=Field(gt=0)
    start:str=Field(min_length=10,max_length=50)
    end:str=Field(min_length=10,max_length=50)
    resources:list[str]=Field(min_length=1,max_length=20)
    reason:str=Field(min_length=1,max_length=300)

def install_scheduling(app,ctx):
    router=APIRouter(prefix='/api/dev/scheduling',dependencies=[Depends(ctx.same_origin_json)])
    @router.get('/appointments')
    def appointments(user=Depends(ctx.user)):
        with ctx.engine.begin() as c:
            if user['role']=='patient':
                rows=c.execute(text("""SELECT a.* FROM dev_appointments a JOIN dev_patients p
                  ON p.id=a.patient_id AND p.tenant=a.tenant
                  WHERE a.tenant=:t AND p.patient_user_id=:u ORDER BY a.start_utc LIMIT 200"""),
                  {'t':user['tenant'],'u':user['id']}).mappings()
            else:
                rows=c.execute(text("""SELECT * FROM dev_appointments
                  WHERE tenant=:t ORDER BY start_utc LIMIT 200"""),{'t':user['tenant']}).mappings()
            return [dict(row) for row in rows]

    @router.post('/appointments',status_code=201)
    def create(body:AppointmentInput,user=Depends(ctx.user)):
        if user['role'] not in ('clinician','org_admin'):
            raise HTTPException(403,'Scheduling staff permission required.')
        try:
            start,end=validate_interval(body.start,body.end)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        if start<=datetime.now(timezone.utc):
            raise HTTPException(422,'Appointment must start in the future.')
        resources=body.resources
        if len(set(resources))!=len(resources) or any(not r.strip() or r!=r.strip() or len(r)>100 for r in resources):
            raise HTTPException(422,'Resources must be unique, nonblank identifiers without surrounding whitespace.')
        start_iso,end_iso=start.isoformat(),end.isoformat()
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            patient=c.execute(text('SELECT 1 FROM dev_patients WHERE id=:p AND tenant=:t'),
                              {'p':body.patient_id,'t':user['tenant']}).scalar()
            if not patient:raise HTTPException(404,'Patient not found.')
            overlap=c.execute(text("""SELECT a.id FROM dev_appointments a
                JOIN dev_appointment_resources r ON r.appointment_id=a.id
                WHERE a.tenant=:t AND a.status='scheduled'
                AND a.start_utc<:end AND a.end_utc>:start
                AND r.resource IN (""" + ','.join(':r'+str(i) for i in range(len(resources))) + """)
                LIMIT 1"""),
                dict(t=user['tenant'],start=start_iso,end=end_iso,
                     **{'r'+str(i):v for i,v in enumerate(resources)})).scalar()
            if overlap:raise HTTPException(409,'A required clinician, room, or equipment resource is already booked.')
            created=c.execute(text("""INSERT INTO dev_appointments
                (tenant,patient_id,created_by,start_utc,end_utc,status,reason,created_at)
                VALUES(:t,:p,:u,:s,:e,'scheduled',:r,:at)"""),
                {'t':user['tenant'],'p':body.patient_id,'u':user['id'],
                 's':start_iso,'e':end_iso,'r':body.reason.strip(),
                 'at':datetime.now(timezone.utc).isoformat()})
            identifier=created.lastrowid
            for resource in resources:
                c.execute(text('INSERT INTO dev_appointment_resources(appointment_id,resource) VALUES(:id,:r)'),
                          {'id':identifier,'r':resource})
            ctx.audit(c,user,'create_synthetic_appointment','appointment',identifier)
        return {'id':identifier,'status':'scheduled','external_booking_confirmed':False}

    @router.post('/appointments/{identifier}/cancel')
    def cancel(identifier:int,user=Depends(ctx.user)):
        if user['role'] not in ('clinician','org_admin'):
            raise HTTPException(403,'Scheduling staff permission required.')
        with ctx.engine.begin() as c:
            c.exec_driver_sql('BEGIN IMMEDIATE')
            result=c.execute(text("""UPDATE dev_appointments SET status='cancelled'
                WHERE id=:id AND tenant=:t AND status='scheduled'"""),
                {'id':identifier,'t':user['tenant']})
            if not result.rowcount:raise HTTPException(404,'Active appointment not found.')
            ctx.audit(c,user,'cancel_synthetic_appointment','appointment',identifier)
        return {'id':identifier,'status':'cancelled','external_notification_sent':False}
    app.include_router(router)
