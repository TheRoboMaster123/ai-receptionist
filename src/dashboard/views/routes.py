from flask import Blueprint, render_template, request, flash, redirect, url_for
from src.dashboard.app import db
from src.dashboard.models.models import Company, VoiceProfile, ResourceUsage
import uuid

bp = Blueprint('dashboard', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/companies')
def list_companies():
    companies = Company.query.all()
    return render_template('companies/list.html', companies=companies)

@bp.route('/companies/new', methods=['GET', 'POST'])
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
        return redirect(url_for('dashboard.list_companies'))
    return render_template('companies/new.html')

@bp.route('/companies/<int:id>')
def view_company(id):
    company = Company.query.get_or_404(id)
    return render_template('companies/view.html', company=company)

@bp.route('/companies/<int:id>/edit', methods=['GET', 'POST'])
def edit_company(id):
    company = Company.query.get_or_404(id)
    if request.method == 'POST':
        company.name = request.form['name']
        company.greeting_message = request.form.get('greeting_message', '')
        company.emergency_contact = request.form.get('emergency_contact', '')
        db.session.commit()
        flash('Company updated successfully!', 'success')
        return redirect(url_for('dashboard.view_company', id=company.id))
    return render_template('companies/edit.html', company=company)

@bp.route('/companies/<int:id>/voice', methods=['GET', 'POST'])
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
        return redirect(url_for('dashboard.view_company', id=company.id))
    return render_template('companies/voice.html', company=company)

@bp.route('/companies/<int:id>/usage')
def view_usage(id):
    company = Company.query.get_or_404(id)
    usage = ResourceUsage.query.filter_by(company_id=id).order_by(ResourceUsage.date.desc()).limit(30).all()
    return render_template('companies/usage.html', company=company, usage=usage) 