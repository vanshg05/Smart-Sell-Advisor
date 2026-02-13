import sys
import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import traceback
import logging
import signal

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log') if os.environ.get('FLASK_ENV') == 'development' else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown handler
def signal_handler(sig, frame):
    logger.info('Shutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

app = Flask(__name__, static_folder='static', template_folder='templates')

# ✅ Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    },
    r"/*": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"]
    }
})

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 
                        'Content-Type, Authorization, Accept, X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 
                        'GET, POST, PUT, DELETE, OPTIONS, PATCH')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    if request.path.startswith('/api/'):
        response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate')
    return response

# Production configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key-123'),
    DEBUG=os.environ.get('FLASK_ENV') == 'development',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload
    JSON_SORT_KEYS=False
)

# Global variables
current_sensor_data = {}
price_forecast_data = {}
ai_models_loaded = False
storage_model = None
price_model = None
scaler = None

# ==================== MODEL COMPATIBILITY FIX ====================
def fix_model_compatibility(model):
    """Fix scikit-learn version compatibility issues"""
    if hasattr(model, 'estimators_'):
        for estimator in model.estimators_:
            if not hasattr(estimator, 'monotonic_cst'):
                estimator.monotonic_cst = None
    return model

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': 'The requested resource was not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.'
    }), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': 'Invalid request parameters'}), 400

# ==================== VALIDATION FUNCTIONS ====================
def validate_sensor_values(temperature, humidity, co2):
    """Validate sensor values and return status with corrections"""
    
    # Define realistic ranges based on wheat storage physics
    VALID_RANGES = {
        'temperature': {'min': -10, 'max': 60, 'optimal_min': 15, 'optimal_max': 35},
        'humidity': {'min': 5, 'max': 100, 'optimal_min': 40, 'optimal_max': 85},
        'co2': {'min': 300, 'max': 20000, 'optimal_min': 400, 'optimal_max': 5000}
    }
    
    issues = []
    corrected_values = {'temperature': temperature, 'humidity': humidity, 'co2': co2}
    extreme_warning = False
    
    # Temperature validation
    if temperature < -20 or temperature > 100:
        issues.append(f"🚨 CRITICAL: Temperature ({temperature}°C) is physically impossible for wheat storage")
        extreme_warning = True
        corrected_values['temperature'] = 25
    elif temperature < VALID_RANGES['temperature']['min']:
        issues.append(f"⚠️ WARNING: Temperature ({temperature}°C) is too low for wheat storage")
        corrected_values['temperature'] = max(VALID_RANGES['temperature']['min'], temperature)
    elif temperature > VALID_RANGES['temperature']['max']:
        issues.append(f"🚨 CRITICAL: Temperature ({temperature}°C) will cause IMMEDIATE spoilage")
        corrected_values['temperature'] = min(VALID_RANGES['temperature']['max'], temperature)
        extreme_warning = True
    elif temperature > 40:
        issues.append(f"⚠️ WARNING: Temperature ({temperature}°C) is dangerously high")
        extreme_warning = True
    
    # Humidity validation
    if humidity < 1 or humidity > 100:
        issues.append(f"🚨 CRITICAL: Humidity ({humidity}%) is physically impossible")
        extreme_warning = True
        corrected_values['humidity'] = 65
    elif humidity < 20:
        issues.append(f"⚠️ WARNING: Humidity ({humidity}%) is too low - wheat will dry out")
        corrected_values['humidity'] = max(40, humidity)
    elif humidity > 95:
        issues.append(f"🚨 CRITICAL: Humidity ({humidity}%) will cause instant mold growth")
        corrected_values['humidity'] = min(85, humidity)
        extreme_warning = True
    elif humidity > 85:
        issues.append(f"⚠️ WARNING: Humidity ({humidity}%) is too high for safe storage")
        extreme_warning = True
    
    # CO2 validation
    if co2 < 200 or co2 > 50000:
        issues.append(f"🚨 CRITICAL: CO₂ ({co2} ppm) indicates sensor malfunction")
        extreme_warning = True
        corrected_values['co2'] = 800
    elif co2 < 350:
        issues.append(f"ℹ️ INFO: CO₂ ({co2} ppm) is unusually low for storage")
    elif co2 > 10000:
        issues.append(f"🚨 CRITICAL: CO₂ ({co2} ppm) indicates severe spoilage")
        extreme_warning = True
        corrected_values['co2'] = min(5000, co2)
    elif co2 > 3000:
        issues.append(f"⚠️ WARNING: CO₂ ({co2} ppm) indicates microbial activity")
        extreme_warning = True
    
    # Check for sensor combination issues
    if temperature > 35 and humidity > 80:
        issues.append("🚨 EXTREME DANGER: High temperature + high humidity = rapid spoilage")
        extreme_warning = True
    
    if temperature > 40 and co2 > 2000:
        issues.append("🚨 CRITICAL: High temp with high CO₂ indicates active spoilage")
        extreme_warning = True
    
    # Calculate validity score
    valid_score = 100
    
    if temperature < VALID_RANGES['temperature']['min'] or temperature > VALID_RANGES['temperature']['max']:
        valid_score -= 40
    elif temperature < VALID_RANGES['temperature']['optimal_min'] or temperature > VALID_RANGES['temperature']['optimal_max']:
        valid_score -= 20
    
    if humidity < VALID_RANGES['humidity']['min'] or humidity > VALID_RANGES['humidity']['max']:
        valid_score -= 40
    elif humidity < VALID_RANGES['humidity']['optimal_min'] or humidity > VALID_RANGES['humidity']['optimal_max']:
        valid_score -= 20
    
    if co2 < VALID_RANGES['co2']['min'] or co2 > VALID_RANGES['co2']['max']:
        valid_score -= 40
    elif co2 < VALID_RANGES['co2']['optimal_min'] or co2 > VALID_RANGES['co2']['optimal_max']:
        valid_score -= 20
    
    validity_status = 'VALID' if valid_score >= 80 else 'QUESTIONABLE' if valid_score >= 60 else 'INVALID'
    
    return {
        'is_valid': validity_status == 'VALID',
        'validity_status': validity_status,
        'validity_score': valid_score,
        'issues': issues,
        'corrected_values': corrected_values,
        'has_extreme_values': extreme_warning,
        'raw_values': {'temperature': temperature, 'humidity': humidity, 'co2': co2}
    }

def handle_extreme_conditions(temperature, humidity, co2):
    """Special handling for extreme conditions"""
    
    emergency_response = {
        'emergency': False,
        'immediate_action': None,
        'spoilage_risk': 'NONE',
        'estimated_spoilage_time': 'UNKNOWN'
    }
    
    # Check for immediate danger conditions
    if temperature >= 45 or humidity >= 90 or co2 >= 5000:
        emergency_response['emergency'] = True
        emergency_response['spoilage_risk'] = 'IMMEDIATE'
        emergency_response['immediate_action'] = 'EVACUATE WHEAT IMMEDIATELY'
        
        if temperature >= 45:
            emergency_response['estimated_spoilage_time'] = '1-3 HOURS'
        elif humidity >= 90:
            emergency_response['estimated_spoilage_time'] = '6-12 HOURS'
        elif co2 >= 5000:
            emergency_response['estimated_spoilage_time'] = '12-24 HOURS'
    
    # Check for high risk conditions
    elif temperature >= 40 or humidity >= 85 or co2 >= 3000:
        emergency_response['emergency'] = True
        emergency_response['spoilage_risk'] = 'HIGH'
        emergency_response['immediate_action'] = 'TAKE CORRECTIVE ACTION WITHIN 24 HOURS'
        emergency_response['estimated_spoilage_time'] = '1-3 DAYS'
    
    # Check for moderate risk
    elif temperature >= 35 or humidity >= 80 or co2 >= 1500:
        emergency_response['spoilage_risk'] = 'MODERATE'
        emergency_response['estimated_spoilage_time'] = '3-7 DAYS'
        emergency_response['immediate_action'] = 'MONITOR CLOSELY AND IMPROVE CONDITIONS'
    
    return emergency_response

# ==================== AI MODEL FUNCTIONS ====================

def load_ai_models():
    """Load trained AI models with fallback - production ready"""
    global ai_models_loaded, storage_model, price_model, scaler
    
    try:
        # Try multiple possible locations for models
        possible_paths = [
            'models/trained/',
            './models/trained/',
            '/opt/render/project/src/models/trained/',
            os.path.join(os.path.dirname(__file__), 'models/trained/')
        ]
        
        models_path = None
        models_loaded = False
        
        for path in possible_paths:
            test_file = os.path.join(path, 'storage_model.pkl')
            if os.path.exists(test_file):
                models_path = path
                logger.info(f"✅ Found models in: {path}")
                break
        
        if not models_path:
            logger.warning("⚠️ AI models not found in any expected location")
            ai_models_loaded = False
            return False
        
        # Load storage model
        storage_model_path = os.path.join(models_path, 'storage_model.pkl')
        if os.path.exists(storage_model_path):
            try:
                with open(storage_model_path, 'rb') as f:
                    storage_model = pickle.load(f)
                storage_model = fix_model_compatibility(storage_model)
                logger.info(f"✅ Storage model loaded successfully from {storage_model_path}")
                models_loaded = True
            except Exception as e:
                logger.error(f"❌ Failed to load storage model: {e}")
                storage_model = None
        else:
            logger.warning(f"⚠️ Storage model not found at {storage_model_path}")
            storage_model = None
        
        # Load price model
        price_model_path = os.path.join(models_path, 'price_model.pkl')
        if os.path.exists(price_model_path):
            try:
                with open(price_model_path, 'rb') as f:
                    price_model = pickle.load(f)
                price_model = fix_model_compatibility(price_model)
                logger.info(f"✅ Price prediction model loaded successfully from {price_model_path}")
                models_loaded = True
            except Exception as e:
                logger.error(f"❌ Failed to load price model: {e}")
                price_model = None
        else:
            logger.warning(f"⚠️ Price model not found at {price_model_path}")
            price_model = None
        
        # Load scaler
        scaler_path = os.path.join(models_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"✅ Scaler loaded successfully from {scaler_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load scaler: {e}")
                scaler = StandardScaler()
        else:
            scaler = StandardScaler()
            logger.warning(f"⚠️ Scaler not found at {scaler_path}, using default")
        
        # Check if we loaded at least one model
        ai_models_loaded = models_loaded
        logger.info(f"🤖 AI Models loaded status: {ai_models_loaded}")
        
        return ai_models_loaded
        
    except Exception as e:
        logger.error(f"❌ Error loading AI models: {e}")
        logger.info("Using rule-based system as fallback")
        ai_models_loaded = False
        return False

def ai_predict_storage(temperature, humidity, co2):
    """Predict safe days using AI model with extreme value handling"""
    global storage_model, scaler
    
    validation = validate_sensor_values(temperature, humidity, co2)
    
    # If values are extreme, use emergency calculation
    if validation['has_extreme_values']:
        logger.warning(f"⚠️ Extreme values detected. Using emergency calculation.")
        return emergency_safe_days_calculation(temperature, humidity, co2)
    
    # If AI model not available, use rule-based
    if not ai_models_loaded or storage_model is None:
        logger.info("⚠️ AI model not available, using rule-based prediction")
        return rule_based_safe_days(temperature, humidity, co2)
    
    try:
        # Prepare input features
        features = np.array([[temperature, humidity, co2]])
        
        # Scale features if scaler exists
        if scaler and hasattr(scaler, 'transform'):
            try:
                features = scaler.transform(features)
            except Exception as e:
                logger.warning(f"⚠️ Feature scaling failed: {e}. Using unscaled features.")
        
        # Predict safe days
        safe_days_pred = storage_model.predict(features)[0]
        
        # Apply validation-based adjustment
        if not validation['is_valid']:
            adjustment_factor = validation['validity_score'] / 100.0
            safe_days_pred *= adjustment_factor
            logger.warning(f"⚠️ Adjusted prediction due to questionable values: {adjustment_factor:.2f}")
        
        # Ensure realistic bounds
        safe_days = max(0.1, min(60, safe_days_pred))
        
        logger.info(f"🤖 AI Prediction: {safe_days:.1f} safe days")
        return safe_days
        
    except Exception as e:
        logger.error(f"❌ AI prediction failed: {e}")
        return rule_based_safe_days(temperature, humidity, co2)

def emergency_safe_days_calculation(temperature, humidity, co2):
    """Calculate safe days for extreme conditions"""
    
    emergency_response = handle_extreme_conditions(temperature, humidity, co2)
    
    if emergency_response['emergency']:
        if emergency_response['spoilage_risk'] == 'IMMEDIATE':
            return 0.1
        elif emergency_response['spoilage_risk'] == 'HIGH':
            return 1.0
        else:
            return 3.0
    
    # Base calculation for extreme but non-emergency
    base_days = 45
    
    # Extreme temperature penalty
    if temperature > 40:
        temp_factor = 0.05
    elif temperature > 35:
        temp_factor = 0.1
    elif temperature > 30:
        temp_factor = 0.3
    elif temperature < 10:
        temp_factor = 0.5
    else:
        temp_factor = 0.7
    
    # Extreme humidity penalty
    if humidity > 85:
        hum_factor = 0.1
    elif humidity > 80:
        hum_factor = 0.3
    elif humidity < 30:
        hum_factor = 0.5
    else:
        hum_factor = 0.7
    
    # Extreme CO2 penalty
    if co2 > 3000:
        co2_factor = 0.2
    elif co2 > 1500:
        co2_factor = 0.5
    else:
        co2_factor = 0.8
    
    safe_days = base_days * temp_factor * hum_factor * co2_factor
    
    return max(0.1, safe_days)

# ==================== RULE-BASED FALLBACK FUNCTIONS ====================

def rule_based_safe_days(temperature, humidity, co2):
    """Rule-based safe days calculation with extreme value handling"""
    
    # First validate inputs
    validation = validate_sensor_values(temperature, humidity, co2)
    
    # If extreme values, use emergency calculation
    if validation['has_extreme_values']:
        logger.warning("⚠️ Extreme values in rule-based calculation")
        return emergency_safe_days_calculation(temperature, humidity, co2)
    
    # Use corrected values if validation flagged issues
    if not validation['is_valid']:
        temp = validation['corrected_values']['temperature']
        hum = validation['corrected_values']['humidity']
        co2_val = validation['corrected_values']['co2']
        logger.info(f"⚠️ Using corrected values: {temp}°C, {hum}%, {co2_val}ppm")
    else:
        temp = temperature
        hum = humidity
        co2_val = co2
    
    base_days = 45
    
    # Temperature effect
    if temp <= 20:
        temp_factor = 1.0
    elif temp <= 25:
        temp_factor = 0.95
    elif temp <= 30:
        temp_factor = np.exp(-0.05 * (temp - 25))
    elif temp <= 35:
        temp_factor = 0.4 * np.exp(-0.1 * (temp - 30))
    else:
        temp_factor = 0.1
    
    # Humidity effect
    if hum <= 65:
        humidity_factor = 1.0
    elif hum <= 70:
        humidity_factor = 0.95
    elif hum <= 75:
        humidity_factor = np.exp(-0.03 * (hum - 70))
    elif hum <= 80:
        humidity_factor = 0.5 * np.exp(-0.05 * (hum - 80))
    else:
        humidity_factor = 0.2
    
    # CO2 effect
    if co2_val <= 800:
        co2_factor = 1.0
    elif co2_val <= 1500:
        co2_factor = 800 / co2_val
    elif co2_val <= 3000:
        co2_factor = 0.5 * (1500 / co2_val)
    else:
        co2_factor = 0.2
    
    safe_days = base_days * temp_factor * humidity_factor * co2_factor
    
    # Apply validation penalty if values were questionable
    if not validation['is_valid']:
        penalty = validation['validity_score'] / 100.0
        safe_days *= penalty
    
    return max(0.5, round(safe_days, 1))

# ==================== MAIN ANALYSIS FUNCTION ====================

def analyze_sensor_data(temperature, humidity, co2, quantity=1000):
    """Main analysis function - AI-enhanced with extreme value handling"""
    
    logger.info(f"🔍 Analyzing: Temp={temperature}°C, Humidity={humidity}%, CO2={co2}ppm")
    
    # Validate sensor values
    validation = validate_sensor_values(temperature, humidity, co2)
    emergency = handle_extreme_conditions(temperature, humidity, co2)
    
    # Calculate confidence score
    confidence = calculate_confidence_score(
        temperature, humidity, co2, 
        validation, emergency, ai_models_loaded
    )
    
    # Log validation results
    if validation['issues']:
        for issue in validation['issues']:
            logger.warning(issue)
    
    # Use AI to predict safe days
    if emergency['emergency']:
        logger.warning(f"🚨 EMERGENCY CONDITION: {emergency['spoilage_risk']} risk")
        safe_days = emergency_safe_days_calculation(temperature, humidity, co2)
        logger.warning(f"🚨 Estimated safe days: {safe_days}")
    else:
        safe_days = ai_predict_storage(temperature, humidity, co2)
    
    # Calculate risk score
    risk_score = calculate_risk_score(temperature, humidity, co2, safe_days)
    
    # Adjust risk score based on validation
    if not validation['is_valid']:
        risk_score = min(100, risk_score + (100 - validation['validity_score']))
    
    risk_level = get_risk_level(risk_score)
    
    # Get recommendations
    recommendations = get_recommendations(temperature, humidity, co2, safe_days, validation, emergency)
    
    # Make decision with AI price forecast
    decision = make_decision(safe_days, risk_level, quantity, emergency)
    
    # Generate detailed report
    detailed_report = generate_detailed_report(
        temperature, humidity, co2, safe_days, risk_level, 
        decision, quantity, validation, emergency
    )
    
    return {
        'sensor_data': {
            'temperature': temperature, 
            'humidity': humidity, 
            'co2': co2,
            'validation': validation,
            'emergency_status': emergency
        },
        'storage_analysis': {
            'safe_days': safe_days,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'recommendations': recommendations,
            'confidence': 0.92 if ai_models_loaded else 0.78,
            'analysis_method': 'AI_ML' if ai_models_loaded else 'Rule_Based',
            'extreme_value_warning': validation['has_extreme_values'],
            'validity_status': validation['validity_status']
        },
        'confidence_score': confidence,
        'risk_score': risk_score,
        'decision': decision,
        'detailed_report': detailed_report,
        'ai_enabled': ai_models_loaded,
        'timestamp': datetime.now().isoformat()
    }

# ==================== CONFIDENCE SCORE CALCULATION ====================

def calculate_confidence_score(temperature, humidity, co2, validation, emergency, ai_models_loaded):
    """Calculate confidence score (0-100) for the analysis"""
    
    confidence = 88.0
    
    if validation['validity_score'] >= 95:
        confidence += 12
    elif validation['validity_score'] >= 90:
        confidence += 8
    elif validation['validity_score'] >= 80:
        confidence += 4
    elif validation['validity_score'] >= 70:
        confidence += 1
    elif validation['validity_score'] < 60:
        confidence -= 8
    
    if validation['has_extreme_values']:
        confidence -= 8
        if len(validation['issues']) > 2:
            confidence -= 3
    
    if emergency['emergency']:
        if emergency['spoilage_risk'] == 'IMMEDIATE':
            confidence -= 20
        elif emergency['spoilage_risk'] == 'HIGH':
            confidence -= 12
        elif emergency['spoilage_risk'] == 'MODERATE':
            confidence -= 6
    
    if ai_models_loaded:
        confidence += 12
    else:
        confidence -= 2
    
    # Optimal ranges
    if 20 <= temperature <= 25:
        confidence += 8
    elif 18 <= temperature <= 28:
        confidence += 4
    elif 15 <= temperature <= 32:
        confidence += 2
    elif 10 <= temperature <= 35:
        confidence += 1
    else:
        confidence -= 2
    
    if 60 <= humidity <= 70:
        confidence += 8
    elif 55 <= humidity <= 75:
        confidence += 4
    elif 50 <= humidity <= 80:
        confidence += 2
    elif 40 <= humidity <= 85:
        confidence += 1
    else:
        confidence -= 2
    
    if 400 <= co2 <= 1000:
        confidence += 8
    elif 350 <= co2 <= 1200:
        confidence += 4
    elif 300 <= co2 <= 1500:
        confidence += 2
    elif 250 <= co2 <= 2000:
        confidence += 1
    else:
        confidence -= 2
    
    # All optimal
    if (20 <= temperature <= 25 and 60 <= humidity <= 70 and 400 <= co2 <= 1000):
        confidence += 10
    
    # Boost low scores
    if confidence < 70 and not emergency['emergency']:
        if confidence >= 60:
            boost = min(70 - confidence, 10)
            confidence += boost
        elif confidence >= 50:
            boost = min(70 - confidence, 15)
            confidence += boost
        else:
            boost = min(70 - confidence, 20)
            confidence += boost
    
    # Good conditions boost
    good_conditions = (
        not emergency['emergency'] and 
        validation['validity_score'] >= 75 and
        15 <= temperature <= 32 and
        50 <= humidity <= 80 and
        300 <= co2 <= 1500
    )
    
    if good_conditions and confidence < 85:
        confidence = max(confidence, 85)
    
    # Cap score
    confidence = max(25, min(96, confidence))
    
    # Determine level
    if confidence >= 90:
        level = "EXCELLENT"
    elif confidence >= 85:
        level = "VERY HIGH"
    elif confidence >= 75:
        level = "HIGH"
    elif confidence >= 65:
        level = "MEDIUM"
    elif confidence >= 50:
        level = "LOW"
    else:
        level = "VERY LOW"
    
    temp_status = "OPTIMAL" if 20 <= temperature <= 25 else "GOOD" if 15 <= temperature <= 30 else "ACCEPTABLE" if 10 <= temperature <= 35 else "POOR"
    humidity_status = "OPTIMAL" if 60 <= humidity <= 70 else "GOOD" if 50 <= humidity <= 80 else "ACCEPTABLE" if 40 <= humidity <= 85 else "POOR"
    co2_status = "OPTIMAL" if 400 <= co2 <= 1000 else "GOOD" if 350 <= co2 <= 1500 else "ACCEPTABLE" if 300 <= co2 <= 2000 else "POOR"
    
    return {
        'score': round(confidence),
        'level': level,
        'breakdown': {
            'validation_impact': validation['validity_score'],
            'ai_model_available': ai_models_loaded,
            'extreme_values': validation['has_extreme_values'],
            'emergency_condition': emergency['emergency'],
            'spoilage_risk': emergency.get('spoilage_risk', 'NONE'),
            'optimal_ranges': {
                'temperature': temp_status,
                'humidity': humidity_status,
                'co2': co2_status
            }
        }
    }

# ==================== ENHANCED HELPER FUNCTIONS ====================

def calculate_risk_score(temperature, humidity, co2, safe_days):
    """Enhanced risk score calculation for extreme values"""
    
    # Temperature risk
    if temperature >= 45:
        temp_risk = 100
    elif temperature >= 40:
        temp_risk = 80 + (temperature - 40) * 4
    elif temperature >= 35:
        temp_risk = 60 + (temperature - 35) * 4
    elif temperature >= 30:
        temp_risk = 30 + (temperature - 30) * 6
    elif temperature >= 25:
        temp_risk = (temperature - 25) * 6
    elif temperature >= 20:
        temp_risk = 0
    elif temperature >= 15:
        temp_risk = 10
    elif temperature >= 5:
        temp_risk = 20 + (15 - temperature)
    else:
        temp_risk = 30
    
    # Humidity risk
    if humidity >= 95:
        humidity_risk = 100
    elif humidity >= 90:
        humidity_risk = 80 + (humidity - 90) * 4
    elif humidity >= 85:
        humidity_risk = 60 + (humidity - 85) * 4
    elif humidity >= 80:
        humidity_risk = 40 + (humidity - 80) * 4
    elif humidity >= 75:
        humidity_risk = 20 + (humidity - 75) * 4
    elif humidity >= 70:
        humidity_risk = (humidity - 70) * 4
    elif humidity >= 60:
        humidity_risk = 0
    elif humidity >= 50:
        humidity_risk = 10
    elif humidity >= 30:
        humidity_risk = 20 + (50 - humidity) * 0.5
    else:
        humidity_risk = 30
    
    # CO2 risk
    if co2 >= 10000:
        co2_risk = 100
    elif co2 >= 5000:
        co2_risk = 80 + (co2 - 5000) / 5000 * 20
    elif co2 >= 3000:
        co2_risk = 60 + (co2 - 3000) / 2000 * 20
    elif co2 >= 1500:
        co2_risk = 40 + (co2 - 1500) / 1500 * 20
    elif co2 >= 1000:
        co2_risk = 20 + (co2 - 1000) / 500 * 20
    elif co2 >= 800:
        co2_risk = (co2 - 800) / 200 * 20
    elif co2 >= 600:
        co2_risk = 0
    elif co2 >= 400:
        co2_risk = 10
    else:
        co2_risk = 20
    
    # Time risk
    if safe_days < 1:
        time_risk = 100
    elif safe_days < 3:
        time_risk = 80 + (3 - safe_days) * 10
    elif safe_days < 7:
        time_risk = 60 + (7 - safe_days) * 5
    elif safe_days < 14:
        time_risk = 30 + (14 - safe_days) * 4.3
    elif safe_days < 21:
        time_risk = (21 - safe_days) * 4.3
    elif safe_days < 30:
        time_risk = (30 - safe_days) * 1.1
    else:
        time_risk = 0
    
    # Combine risks
    weights = {'temp': 0.3, 'humidity': 0.3, 'co2': 0.2, 'time': 0.2}
    
    if temperature >= 35 or temperature <= 10:
        weights['temp'] = 0.4
        weights['time'] = 0.1
    
    if humidity >= 80 or humidity <= 40:
        weights['humidity'] = 0.4
        weights['co2'] = 0.1
    
    total_risk = (
        min(100, temp_risk) * weights['temp'] +
        min(100, humidity_risk) * weights['humidity'] +
        min(100, co2_risk) * weights['co2'] +
        min(100, time_risk) * weights['time']
    )
    
    return round(total_risk, 1)

def get_risk_level(risk_score):
    """Convert risk score to risk level"""
    if risk_score >= 70:
        return 'CRITICAL'
    elif risk_score >= 50:
        return 'HIGH'
    elif risk_score >= 30:
        return 'MODERATE'
    else:
        return 'LOW'

def get_recommendations(temperature, humidity, co2, safe_days, validation=None, emergency=None):
    """Enhanced recommendations for extreme values"""
    recommendations = []
    
    if validation and validation['issues']:
        for issue in validation['issues']:
            if 'CRITICAL' in issue or '🚨' in issue:
                recommendations.append(f"🚨 {issue}")
            else:
                recommendations.append(f"⚠️ {issue}")
    
    if emergency and emergency['emergency']:
        recommendations.append(f"🚨 EMERGENCY: {emergency['immediate_action']}")
        recommendations.append(f"🚨 Estimated spoilage time: {emergency['estimated_spoilage_time']}")
    
    # Temperature recommendations
    if temperature >= 45:
        recommendations.append("🚨 IMMEDIATE ACTION: Temperature is lethal for wheat")
        recommendations.append("🚨 Move wheat to cool storage or sell immediately")
    elif temperature >= 40:
        recommendations.append("🚨 CRITICAL: Temperature will cause rapid spoilage")
        recommendations.append("🚨 Increase ventilation and consider emergency sale")
    elif temperature >= 35:
        recommendations.append("⚠️ DANGER: High temperature accelerating spoilage")
        recommendations.append("⚠️ Install cooling system or reduce storage density")
    elif temperature >= 30:
        recommendations.append("⚠️ Reduce storage temperature below 28°C")
    elif temperature <= 10:
        recommendations.append("⚠️ Temperature too low - risk of condensation")
    elif 20 <= temperature <= 25:
        recommendations.append("✅ Temperature is optimal for wheat storage")
    
    # Humidity recommendations
    if humidity >= 90:
        recommendations.append("🚨 IMMEDIATE ACTION: Humidity will cause instant mold")
        recommendations.append("🚨 Dehumidify immediately or sell all wheat")
    elif humidity >= 85:
        recommendations.append("🚨 CRITICAL: Very high humidity causing mold growth")
        recommendations.append("🚨 Increase ventilation and use dehumidifiers")
    elif humidity >= 80:
        recommendations.append("⚠️ DANGER: High humidity promoting fungal growth")
        recommendations.append("⚠️ Improve ventilation to reduce humidity")
    elif humidity <= 30:
        recommendations.append("⚠️ Very low humidity - wheat may dry out excessively")
    elif 60 <= humidity <= 70:
        recommendations.append("✅ Humidity is within optimal range")
    
    # CO2 recommendations
    if co2 >= 10000:
        recommendations.append("🚨 IMMEDIATE ACTION: Extreme CO2 indicates severe spoilage")
        recommendations.append("🚨 Inspect wheat immediately - may already be spoiled")
    elif co2 >= 5000:
        recommendations.append("🚨 CRITICAL: Very high CO2 indicates active spoilage")
        recommendations.append("🚨 Aerate storage immediately and inspect wheat")
    elif co2 >= 3000:
        recommendations.append("⚠️ DANGER: High CO2 indicates microbial activity")
        recommendations.append("⚠️ Increase ventilation and monitor closely")
    
    # Time-based recommendations
    if safe_days < 1:
        recommendations.append("🚨 IMMEDIATE: Less than 1 day of safe storage remaining")
        recommendations.append("🚨 Sell or process wheat immediately")
    elif safe_days < 3:
        recommendations.append("🚨 CRITICAL: Very limited safe storage days remaining")
        recommendations.append("🚨 Prepare for immediate sale")
    elif safe_days < 7:
        recommendations.append("⚠️ Limited safe storage days remaining")
        recommendations.append("⚠️ Arrange sale within this week")
    
    if not recommendations:
        recommendations.append("✅ Storage conditions are optimal")
    
    return recommendations[:10]

def make_decision(safe_days, risk_level, quantity, emergency=None):
    """Enhanced decision making for extreme conditions"""
    global price_forecast_data
    
    # Check if price_forecast_data exists
    if not price_forecast_data or 'statistics' not in price_forecast_data:
        price_stats = {
            'max_expected_price': 2200,
            'optimal_sell_day': 7,
            'average_expected_price': 2100,
            'trend': 'STABLE'
        }
    else:
        price_stats = price_forecast_data['statistics']
    
    max_price = price_stats['max_expected_price']
    optimal_day = price_stats['optimal_sell_day']
    avg_price = price_stats['average_expected_price']
    trend = price_stats['trend']
    
    # Emergency override
    if emergency and emergency['emergency']:
        if emergency['spoilage_risk'] == 'IMMEDIATE':
            return {
                'action': 'SELL IMMEDIATELY - EMERGENCY',
                'reason': emergency['immediate_action'],
                'recommended_action_day': 0,
                'expected_price_per_quintal': round(max_price * 0.8, 2),
                'expected_total_value': round(max_price * quantity * 0.8 / 100, 2),
                'confidence': 'VERY HIGH',
                'decision_score': 1.0,
                'emergency': True
            }
        elif emergency['spoilage_risk'] == 'HIGH':
            return {
                'action': 'SELL URGENTLY',
                'reason': f"High spoilage risk: {emergency['estimated_spoilage_time']}",
                'recommended_action_day': 1,
                'expected_price_per_quintal': round(max_price * 0.85, 2),
                'expected_total_value': round(max_price * quantity * 0.85 / 100, 2),
                'confidence': 'HIGH',
                'decision_score': 0.9,
                'emergency': True
            }
    
    # Normal decision logic
    price_score = max_price / avg_price if avg_price > 0 else 1.0
    risk_score = 1.0 if risk_level == 'LOW' else 0.7 if risk_level == 'MODERATE' else 0.4 if risk_level == 'HIGH' else 0.1
    time_score = min(1.0, safe_days / 30)
    
    decision_score = (price_score * 0.4) + (risk_score * 0.4) + (time_score * 0.2)
    
    if risk_level == "CRITICAL" or safe_days < 1:
        action = "SELL IMMEDIATELY"
        reason = "Critical storage conditions detected"
        recommended_day = 0
        confidence = "VERY HIGH"
    elif safe_days < 3:
        action = "SELL TODAY"
        reason = f"Very limited safe storage ({safe_days} days)"
        recommended_day = 0
        confidence = "HIGH"
    elif decision_score > 0.8 and optimal_day <= 7:
        action = "SELL NOW"
        reason = f"Excellent selling opportunity - best price in {optimal_day} days"
        recommended_day = optimal_day
        confidence = "HIGH"
    elif decision_score > 0.6 and optimal_day <= safe_days:
        action = "SELL SOON"
        reason = f"Good selling window - optimal price in {optimal_day} days"
        recommended_day = optimal_day
        confidence = "MEDIUM"
    elif safe_days >= 21:
        action = "CONTINUE STORING"
        reason = f"Ample storage time ({safe_days} days). Market trend: {trend.lower()}"
        recommended_day = min(optimal_day, safe_days)
        confidence = "HIGH"
    else:
        action = "HOLD AND MONITOR"
        reason = f"Monitor for {min(7, safe_days)} days. Market is {trend.lower()}"
        recommended_day = min(7, safe_days)
        confidence = "MEDIUM"
    
    expected_value = max_price * quantity / 100
    
    return {
        'action': action,
        'reason': reason,
        'recommended_action_day': recommended_day,
        'expected_price_per_quintal': round(max_price, 2),
        'expected_total_value': round(expected_value, 2),
        'confidence': confidence,
        'decision_score': round(decision_score, 2),
        'emergency': False
    }

def generate_detailed_report(temperature, humidity, co2, safe_days, risk_level, 
                           decision, quantity, validation, emergency):
    """Generate detailed analysis report with extreme value warnings"""
    
    risk_score = calculate_risk_score(temperature, humidity, co2, safe_days)
    
    # Check if price_forecast_data exists
    if price_forecast_data and 'statistics' in price_forecast_data:
        price_stats = price_forecast_data['statistics']
        max_price = price_stats['max_expected_price']
        avg_price = price_stats['average_expected_price']
        trend = price_stats['trend']
        min_price = price_stats.get('min_expected_price', max_price * 0.9)
        optimal_day = price_stats.get('optimal_sell_day', 7)
    else:
        max_price = 2200
        avg_price = 2100
        trend = "STABLE"
        min_price = 2000
        optimal_day = 7
    
    method = "AI/ML" if ai_models_loaded else "Rule-Based"
    
    report = f"""WHEAT STORAGE ANALYSIS REPORT
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Method: {method}
Data Validity: {validation['validity_status']} ({validation['validity_score']}/100)

{"="*60}
🚨 {'EMERGENCY ALERT' if emergency['emergency'] else 'NORMAL CONDITIONS'}
{"="*60}
"""

    if emergency['emergency']:
        report += f"""
EMERGENCY STATUS:
• Spoilage Risk: {emergency['spoilage_risk']}
• Immediate Action Required: {emergency['immediate_action']}
• Estimated Time to Spoilage: {emergency['estimated_spoilage_time']}
"""
    
    report += f"""
CURRENT STORAGE CONDITIONS:
• Temperature: {temperature}°C {'(EXTREME)' if validation['has_extreme_values'] else '(Optimal)' if 20 <= temperature <= 25 else '(High)' if temperature > 25 else '(Low)'}
• Humidity: {humidity}% {'(EXTREME)' if validation['has_extreme_values'] else '(Optimal)' if 60 <= humidity <= 70 else '(High)' if humidity > 70 else '(Low)'}
• CO₂ Level: {co2} PPM {'(EXTREME)' if validation['has_extreme_values'] else '(Normal)' if co2 < 1000 else '(Elevated)' if co2 < 1500 else '(High)'}

DATA VALIDATION NOTES:"""
    
    if validation['issues']:
        for issue in validation['issues']:
            report += f"\n• {issue}"
    else:
        report += "\n• All sensor readings within expected ranges"
    
    report += f"""

STORAGE HEALTH ASSESSMENT:
• Safe Storage Days Remaining: {safe_days} days {'(CRITICAL)' if safe_days < 3 else '(LOW)' if safe_days < 7 else '(ADEGUATE)' if safe_days < 14 else '(GOOD)'}
• Risk Level: {risk_level}
• Risk Score: {risk_score}/100

PRICE FORECAST:
• Current Market Trend: {trend}
• Expected Price Range: ₹{min_price} - ₹{max_price}/quintal
• Optimal Selling Day: Day {optimal_day}
• Average Expected Price: ₹{avg_price}/quintal

FINANCIAL PROJECTION:
• Wheat Quantity: {quantity} quintals
• Expected Sale Value: ₹{decision['expected_total_value']:,.2f}
• Expected Price per Quintal: ₹{decision['expected_price_per_quintal']}

{'='*60}
{"🚨 EMERGENCY RECOMMENDATION" if emergency['emergency'] else "RECOMMENDATION"}
{'='*60}
Action: {decision['action']}
Confidence Level: {decision['confidence']}
Decision Score: {decision['decision_score']}/1.0

REASONING: {decision['reason']}

RECOMMENDED ACTIONS:"""
    
    recommendations = get_recommendations(temperature, humidity, co2, safe_days, validation, emergency)
    for i, rec in enumerate(recommendations[:5], 1):
        report += f"\n{i}. {rec}"
    
    report += f"""

IMMEDIATE PRIORITIES:
1. {'🚨 ' if emergency['emergency'] else ''}{'Immediately contact buyers and prepare wheat for transport' if 'SELL' in decision['action'] else 'Continue monitoring storage conditions daily'}
2. {'🚨 ' if validation['has_extreme_values'] else ''}{'Verify sensor readings - possible malfunction' if validation['has_extreme_values'] else 'Check temperature and humidity twice daily'}
3. {'🚨 ' if safe_days < 7 else ''}{'Arrange alternative storage or immediate sale' if safe_days < 7 else 'Maintain current storage conditions'}

NEXT REVIEW: In {min(1 if emergency['emergency'] else 3, safe_days)} days or if conditions change significantly.

Note: This analysis includes validation of sensor readings. 
{'🚨 EXTREME VALUES DETECTED - Verify sensor accuracy immediately.' if validation['has_extreme_values'] else 'All readings appear valid.'}
"""
    
    return report

# ==================== PRICE FORECAST FUNCTIONS ====================

def ai_predict_price_forecast():
    """Generate price forecast using AI model with fallback"""
    global price_model
    
    if not ai_models_loaded or price_model is None:
        logger.info("⚠️ AI price model not available, using rule-based forecast")
        return rule_based_price_forecast()
    
    try:
        # Try to load price data
        price_data_paths = [
            'price_data.csv',
            './price_data.csv',
            'data/price_data.csv',
            os.path.join(os.path.dirname(__file__), 'price_data.csv')
        ]
        
        price_data = None
        for path in price_data_paths:
            if os.path.exists(path):
                try:
                    price_data = pd.read_csv(path)
                    logger.info(f"✅ Loaded price data from: {path}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load price data from {path}: {e}")
        
        if price_data is None or len(price_data) < 30:
            logger.warning("⚠️ Insufficient price data, using rule-based forecast")
            return rule_based_price_forecast()
        
        # Prepare features for prediction
        if 'p_modal' in price_data.columns:
            recent_prices = price_data['p_modal'].tail(30).values
        elif 'price' in price_data.columns:
            recent_prices = price_data['price'].tail(30).values
        else:
            recent_prices = price_data.iloc[:, -1].tail(30).values
        
        # Generate 30-day forecast
        predictions = []
        current_features = recent_prices.copy()
        
        for _ in range(30):
            X = current_features.reshape(1, -1)
            next_price = price_model.predict(X)[0]
            predictions.append(next_price)
            current_features = np.roll(current_features, -1)
            current_features[-1] = next_price
        
        # Generate dates
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(30)]
        
        # Calculate statistics
        predictions = np.array(predictions)
        optimal_day = np.argmax(predictions) + 1
        
        forecast_data = {
            'predictions': predictions.tolist(),
            'dates': dates,
            'statistics': {
                'optimal_sell_day': int(optimal_day),
                'max_expected_price': round(float(np.max(predictions)), 2),
                'min_expected_price': round(float(np.min(predictions)), 2),
                'average_expected_price': round(float(np.mean(predictions)), 2),
                'trend': "UPWARD" if predictions[-1] > predictions[0] else "DOWNWARD",
                'volatility_percentage': round(np.std(predictions) / np.mean(predictions) * 100, 2)
            }
        }
        
        logger.info(f"🤖 AI Price Forecast generated: {len(predictions)} days")
        return forecast_data
        
    except Exception as e:
        logger.error(f"❌ AI price prediction failed: {e}")
        return rule_based_price_forecast()

def rule_based_price_forecast():
    """Rule-based price forecast (Fallback)"""
    base_price = 2200
    predictions = []
    dates = []
    
    for i in range(30):
        date = datetime.now() + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        
        trend = base_price * (1 + i * 0.001)
        seasonal = 100 * np.sin(2 * np.pi * i / 30)
        random_comp = np.random.normal(0, 30)
        
        price = trend + seasonal + random_comp
        predictions.append(max(1800, price))
    
    optimal_day = np.argmax(predictions) + 1
    
    forecast_data = {
        'predictions': [float(p) for p in predictions],
        'dates': dates,
        'statistics': {
            'optimal_sell_day': int(optimal_day),
            'max_expected_price': round(float(max(predictions)), 2),
            'min_expected_price': round(float(min(predictions)), 2),
            'average_expected_price': round(float(np.mean(predictions)), 2),
            'trend': "UPWARD" if predictions[-1] > predictions[0] else "DOWNWARD",
            'volatility_percentage': round(np.std(predictions) / np.mean(predictions) * 100, 2)
        }
    }
    
    logger.info(f"📊 Rule-based Price Forecast generated: {len(predictions)} days")
    return forecast_data

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    """Home page - serve dashboard"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home: {e}")
        return """
        <html>
            <head><title>Smart Sell Advisor</title></head>
            <body>
                <h1>Smart Sell Advisor - AI Wheat Storage System</h1>
                <p>Service is running. API endpoints are available at /api/*</p>
                <p><a href="/api/health">Health Check</a></p>
                <p><a href="/api/price-forecast">Price Forecast</a></p>
            </body>
        </html>
        """

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return jsonify({'error': 'Dashboard template not found'}), 404

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'Smart Sell Advisor AI - Enhanced',
            'timestamp': datetime.now().isoformat(),
            'version': '2.2.0',
            'ai_enabled': ai_models_loaded,
            'extreme_value_handling': True,
            'environment': os.environ.get('FLASK_ENV', 'development'),
            'python_version': sys.version,
            'models_loaded': {
                'storage': storage_model is not None,
                'price': price_model is not None,
                'scaler': scaler is not None
            }
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/current-status')
def get_current_status():
    """Get current sensor data status"""
    if not current_sensor_data:
        return jsonify({
            'status': 'no_data',
            'message': 'No sensor data available. Use manual input.',
            'timestamp': datetime.now().isoformat(),
            'sample_data_available': True
        })
    
    try:
        analysis = analyze_sensor_data(
            current_sensor_data.get('temperature', 25),
            current_sensor_data.get('humidity', 65),
            current_sensor_data.get('co2', 800)
        )
        analysis['price_forecast'] = price_forecast_data
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error in current-status: {e}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/sensor-data', methods=['POST', 'OPTIONS'])
def receive_sensor_data():
    """Receive sensor data from ESP8266 or other devices"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        data = request.json
        logger.info(f"📥 Received data from device: {data}")
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        # Check for required fields
        required_fields = ['temperature', 'humidity', 'co2']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing sensor data: {", ".join(missing_fields)}',
                'received_fields': list(data.keys())
            }), 400
        
        # Extract and validate values
        try:
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            co2 = float(data['co2'])
        except ValueError as e:
            return jsonify({
                'error': 'Invalid data format',
                'message': 'Temperature, humidity, and CO2 must be numeric values',
                'received': data
            }), 400
        
        # Validate before storing
        validation = validate_sensor_values(temperature, humidity, co2)
        emergency = handle_extreme_conditions(temperature, humidity, co2)
        
        global current_sensor_data
        current_sensor_data = {
            'temperature': temperature,
            'humidity': humidity,
            'co2': co2,
            'received_at': datetime.now().isoformat(),
            'source': data.get('source', 'device'),
            'device_id': data.get('device_id', 'unknown_device'),
            'validation': validation,
            'emergency': emergency
        }
        
        logger.info(f"✅ Data stored: Temp={temperature}°C, Humidity={humidity}%, CO2={co2}ppm")
        
        if validation['has_extreme_values']:
            logger.warning(f"⚠️ Extreme values detected in sensor data")
            for issue in validation['issues']:
                logger.warning(f"   {issue}")
        
        if emergency['emergency']:
            logger.error(f"🚨 EMERGENCY CONDITION: {emergency['spoilage_risk']} risk!")

        response = {
            'status': 'success',
            'message': 'Data received successfully',
            'timestamp': datetime.now().isoformat(),
            'received_data': {
                'temperature': temperature,
                'humidity': humidity,
                'co2': co2
            },
            'validation': validation,
            'emergency': emergency,
            'analysis_available': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error receiving sensor data: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/manual-input', methods=['POST'])
def manual_input():
    """Manual input endpoint for testing"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        required = ['temperature', 'humidity', 'co2', 'wheat_quantity']
        missing_fields = [field for field in required if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Convert values with error handling
        try:
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            co2 = float(data['co2'])
            quantity = float(data['wheat_quantity'])
        except ValueError as e:
            return jsonify({
                'error': 'Invalid numeric values',
                'message': 'All fields must be numeric'
            }), 400
        
        # Validate ranges
        if temperature < -50 or temperature > 100:
            return jsonify({'error': 'Temperature outside possible range (-50 to 100°C)'}), 400
        if humidity < 0 or humidity > 100:
            return jsonify({'error': 'Humidity must be 0-100%'}), 400
        if co2 < 0 or co2 > 100000:
            return jsonify({'error': 'CO2 must be 0-100000 PPM'}), 400
        if quantity <= 0 or quantity > 1000000:
            return jsonify({'error': 'Wheat quantity must be between 0 and 1,000,000 quintals'}), 400
        
        # Perform analysis
        analysis = analyze_sensor_data(temperature, humidity, co2, quantity)
        analysis['price_forecast'] = price_forecast_data
        
        # Store as current
        global current_sensor_data
        validation = validate_sensor_values(temperature, humidity, co2)
        emergency = handle_extreme_conditions(temperature, humidity, co2)
        
        current_sensor_data = {
            'temperature': temperature,
            'humidity': humidity,
            'co2': co2,
            'wheat_quantity': quantity,
            'received_at': datetime.now().isoformat(),
            'source': 'manual',
            'validation': validation,
            'emergency': emergency
        }
        
        logger.info(f"📝 Manual analysis completed for Temp={temperature}°C, Hum={humidity}%, CO2={co2}ppm")
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error in manual-input: {e}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500

@app.route('/api/price-forecast')
def get_price_forecast():
    """Get price forecast"""
    try:
        return jsonify(price_forecast_data)
    except Exception as e:
        logger.error(f"Error getting price forecast: {e}")
        return jsonify({
            'error': 'Price forecast unavailable',
            'message': str(e),
            'fallback': rule_based_price_forecast()
        })

@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Reset current data"""
    global current_sensor_data, price_forecast_data
    current_sensor_data = {}
    price_forecast_data = ai_predict_price_forecast()
    logger.info("🔄 Data reset successfully")
    return jsonify({
        'status': 'success', 
        'message': 'Data reset successfully',
        'price_forecast_regenerated': True
    })

@app.route('/api/test-extreme-values', methods=['POST'])
def test_extreme_values():
    """Test endpoint for extreme value handling"""
    try:
        test_cases = [
            {'temperature': 55, 'humidity': 20, 'co2': 10000, 'name': 'Your Example'},
            {'temperature': -5, 'humidity': 95, 'co2': 500, 'name': 'Freezing + High Humidity'},
            {'temperature': 60, 'humidity': 10, 'co2': 20000, 'name': 'Extreme All'},
            {'temperature': 25, 'humidity': 65, 'co2': 800, 'name': 'Normal'},
            {'temperature': 45, 'humidity': 85, 'co2': 5000, 'name': 'Danger Zone'}
        ]
        
        results = []
        
        for test in test_cases:
            logger.info(f"🧪 Testing: {test['name']}")
            validation = validate_sensor_values(test['temperature'], test['humidity'], test['co2'])
            emergency = handle_extreme_conditions(test['temperature'], test['humidity'], test['co2'])
            safe_days = ai_predict_storage(test['temperature'], test['humidity'], test['co2'])
            
            results.append({
                'test_case': test['name'],
                'values': test,
                'validation': validation,
                'emergency': emergency,
                'safe_days_prediction': safe_days,
                'analysis_method': 'AI' if ai_models_loaded and storage_model else 'Rule-Based'
            })
        
        return jsonify({
            'status': 'success',
            'tests': results,
            'timestamp': datetime.now().isoformat(),
            'ai_enabled': ai_models_loaded,
            'total_tests': len(test_cases)
        })
        
    except Exception as e:
        logger.error(f"Error in test-extreme-values: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-info')
def system_info():
    """System information endpoint"""
    import platform
    
    return jsonify({
        'system': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'hostname': platform.node()
        },
        'application': {
            'ai_models_loaded': ai_models_loaded,
            'storage_model': storage_model is not None,
            'price_model': price_model is not None,
            'current_data': bool(current_sensor_data),
            'price_forecast': bool(price_forecast_data),
            'environment': os.environ.get('FLASK_ENV', 'development')
        },
        'timestamp': datetime.now().isoformat()
    })

# ==================== INITIALIZATION ====================

def init_app():
    """Initialize application with AI models"""
    global current_sensor_data, price_forecast_data
    
    print("\n" + "="*70)
    print(" SMART SELL ADVISOR - AI Enhanced with Extreme Value Handling")
    print(" Version: 2.2.0 - Production Ready")
    print("="*70)
    
    # Load AI models
    ai_loaded = load_ai_models()
    
    if ai_loaded:
        print("🤖 AI Models: LOADED")
    else:
        print("🤖 AI Models: NOT LOADED (Using rule-based fallback)")
    
    # Initialize sample data
    current_sensor_data = {
        'temperature': 25.5,
        'humidity': 65.0,
        'co2': 850,
        'received_at': datetime.now().isoformat(),
        'source': 'sample'
    }
    
    # Generate price forecast
    price_forecast_data = ai_predict_price_forecast()
    
    print(f"💰 Price Forecast: Generated ({len(price_forecast_data['predictions'])} days)")
    print(f"⚙️  Analysis Method: {'AI/ML' if ai_loaded else 'Rule-Based'}")
    print(f"🛡️  Extreme Value Handling: ENABLED")
    print(f"🌐 Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"📊 Memory: Initialized with sample data")
    print("="*70 + "\n")
    
    # Test extreme value handling
    print("🧪 Testing Extreme Value Handling...")
    test_values = [
        (55, 20, 10000),
        (-5, 95, 500),
        (60, 10, 20000)
    ]
    
    for temp, hum, co2 in test_values:
        validation = validate_sensor_values(temp, hum, co2)
        status = validation['validity_status']
        issues = len(validation['issues'])
        print(f"   {temp}°C, {hum}%, {co2}ppm → {status} ({issues} issues)")

# Initialize when starting
if __name__ == '__main__':
    init_app()
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"\n{'='*70}")
    print(f" Server starting on port: {port}")
    print(f" Debug mode: {debug_mode}")
    print(f" AI Enabled: {ai_models_loaded}")
    print(f" Extreme Value Handling: ENABLED")
    print(f"{'='*70}")
    
    if os.environ.get('FLASK_ENV') == 'production':
        print("🚀 Running in production mode...")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        print("🔧 Running in development mode...")
        app.run(host='0.0.0.0', port=port, debug=True)
else:
    # For Gunicorn deployment
    init_app()