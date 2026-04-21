from django.test import TestCase

from .views import get_patient_suggestions


class PatientSuggestionTests(TestCase):
    def test_positive_prediction_suggestions_include_consultation_guidance(self):
        suggestions = get_patient_suggestions(True)

        self.assertEqual(suggestions['title'], 'Suggested next steps')
        self.assertIn('doctor', suggestions['disclaimer'].lower())

    def test_negative_prediction_suggestions_include_prevention_guidance(self):
        suggestions = get_patient_suggestions(False)

        self.assertEqual(suggestions['title'], 'Suggested health guidance')
        self.assertTrue(any('checkup' in item.lower() for item in suggestions['items']))
