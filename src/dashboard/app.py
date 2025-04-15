from flask import Flask, render_template, request, flash, redirect, url_for
from pathlib import Path
import uuid

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dashboard.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Import and initialize models
from src.dashboard.models.models import db, Company, VoiceProfile, ResourceUsage
db.init_app(app)

# Basic routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/companies')
def list_companies():
    companies = Company.query.all()
    return render_template('companies/list.html', companies=companies)

@app.route('/companies/new', methods=['GET', 'POST'])
def new_company():
    if request.method == 'POST':
        company = Company(
            name=request.form['name'],
            api_key=str(uuid.uuid4()),
            business_hours={},
            greeting_message=request.form.get('greeting_message', '')
        )
        db.session.add(company)
        db.session.commit()
        flash('Company created successfully!', 'success')
        return redirect(url_for('list_companies'))
    return render_template('companies/new.html')

@app.route('/companies/<int:id>')
def view_company(id):
    company = Company.query.get_or_404(id)
    return render_template('companies/view.html', company=company)

@app.route('/companies/<int:id>/edit', methods=['GET', 'POST'])
def edit_company(id):
    company = Company.query.get_or_404(id)
    if request.method == 'POST':
        company.name = request.form['name']
        company.greeting_message = request.form.get('greeting_message', '')
        company.emergency_contact = request.form.get('emergency_contact', '')
        db.session.commit()
        flash('Company updated successfully!', 'success')
        return redirect(url_for('view_company', id=company.id))
    return render_template('companies/edit.html', company=company)

@app.route('/companies/<int:id>/voice', methods=['GET', 'POST'])
def manage_voice(id):
    company = Company.query.get_or_404(id)
    if request.method == 'POST':
        if not company.voice_profile:
            voice_profile = VoiceProfile(company_id=company.id)
            db.session.add(voice_profile)
        else:
            voice_profile = company.voice_profile
        
        voice_profile.voice_settings = request.form.get('voice_settings', '{}')
        if 'voice_model' in request.files:
            file = request.files['voice_model']
            if file:
                # TODO: Handle voice model file upload
                pass
        
        db.session.commit()
        flash('Voice settings updated successfully!', 'success')
        return redirect(url_for('view_company', id=company.id))
    return render_template('companies/voice.html', company=company)

@app.route('/health')
def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False) 