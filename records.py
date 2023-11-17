import models

import sqlite3
import os

database_file_path = None
def initialize(args, project_name: str):
    global database_file_path
    database_file_path = os.path.join(args.results_directory, project_name + '.db')

    conn = sqlite3.connect(database_file_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS models (
        name TEXT PRIMARY KEY,
        origin_job_id INTEGER NOT NULL,
        origin_job_suite TEXT NOT NULL,
        training_dataset TEXT NOT NULL,
        architecture TEXT NOT NULL,
        
        checkpoint_path TEXT,
        training_epochs INTEGER,
        training_loss REAL,
        training_accuracy REAL,
        validation_loss REAL,
        validation_accuracy REAL
    )
    ''')
    c.execute('''CREATE TABLE IF NOT EXISTS evaluations (
        job_id INTEGER PRIMARY KEY,
        job_suite TEXT NOT NULL,
        model_name TEXT NOT NULL,
        dataset TEXT NOT NULL,
        split TEXT NOT NULL,
        
        loss REAL,
        accuracy REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS jobs (
        job_id INTEGER PRIMARY KEY,
        job_suite TEXT NOT NULL,
        action TEXT NOT NULL,
        dataset TEXT NOT NULL,
        split TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        last_status TEXT DEFAULT 'started'
    )''')

    conn.commit()
    conn.close()

def insert_model(args, model_name: str, epoch_age: int, training_loss: float, training_accuracy: float, validation_loss: float, validation_accuracy: float):
    if database_file_path is None:
        raise ValueError('Database not initialized. Call records.initialize() first.')
    
    conn = sqlite3.connect(database_file_path)
    c = conn.cursor()

    values = {
        'name': model_name,
        'origin_job_id': args.job_id,
        'origin_job_suite': args.job_suite,
        'training_dataset': args.dataset,
        'architecture': args.architecture,
        'checkpoint_path': models.make_checkpoint_path(args, model_name),
        'training_epochs': epoch_age,
        'training_loss': training_loss,
        'training_accuracy': training_accuracy,
        'validation_loss': validation_loss,
        'validation_accuracy': validation_accuracy
    }
    c.execute('''INSERT INTO models (
        name, origin_job_id, origin_job_suite, training_dataset, architecture,
        checkpoint_path, training_epochs, training_loss, training_accuracy, validation_loss, validation_accuracy
    ) VALUES (
        :name, :origin_job_id, :origin_job_suite, :training_dataset, :architecture,
        :checkpoint_path, :training_epochs, :training_loss, :training_accuracy, :validation_loss, :validation_accuracy
    ) ON CONFLICT (name) DO UPDATE SET
        training_epochs=excluded.training_epochs,
        training_loss=excluded.training_loss,
        training_accuracy=excluded.training_accuracy,
        validation_loss=excluded.validation_loss,
        validation_accuracy=excluded.validation_accuracy
    ''', {
        'name': model_name,
        'origin_job_id': args.job_id,
        'origin_job_suite': args.job_suite,
        'training_dataset': args.dataset,
        'architecture': args.architecture,
        'checkpoint_path': models.make_checkpoint_path(args, model_name),
        'training_epochs': epoch_age,
        'training_loss': training_loss,
        'training_accuracy': training_accuracy,
        'validation_loss': validation_loss,
        'validation_accuracy': validation_accuracy
    })
    conn.commit()
    conn.close()

def insert_evaluation(args, model_name, loss, accuracy):
    if database_file_path is None:
        raise ValueError('Database not initialized. Call records.initialize() first.')
    
    conn = sqlite3.connect(database_file_path)
    c = conn.cursor()
    c.execute('''INSERT INTO evaluations (
        job_id, job_suite, model_name, dataset, split, loss, accuracy
    ) VALUES (
        :job_id, :job_suite, :model_name, :dataset, :split, :loss, :accuracy
    )''', {
        'job_id': args.job_id,
        'job_suite': args.job_suite,
        'model_name': model_name,
        'dataset': args.dataset,
        'split': args.split,
        'loss': loss,
        'accuracy': accuracy
    })
    conn.commit()
    conn.close()

def insert_job(args):
    if database_file_path is None:
        raise ValueError('Database not initialized. Call records.initialize() first.')
    
    conn = sqlite3.connect(database_file_path)
    c = conn.cursor()
    c.execute('''INSERT INTO jobs (
        job_id, job_suite, action, dataset, split, timestamp
    ) VALUES (
        :job_id, :job_suite, :action, :dataset, :split, :timestamp
    )''', {
        'job_id': args.job_id,
        'job_suite': args.job_suite,
        'action': args.action,
        'dataset': args.dataset,
        'split': args.split,
        'timestamp': args.job_id
    })
    conn.commit()
    conn.close()

def update_job_status(args, new_status):
    if database_file_path is None:
        raise ValueError('Database not initialized. Call records.initialize() first.')
    
    conn = sqlite3.connect(database_file_path)
    c = conn.cursor()
    c.execute('''UPDATE jobs SET last_status=:new_status WHERE job_id=:job_id''', {
        'job_id': args.job_id,
        'new_status': new_status
    })
    conn.commit()
    conn.close()