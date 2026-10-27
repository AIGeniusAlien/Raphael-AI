"""Claim-screen protocol tests use model fixtures,not medical accuracy evidence."""
import json
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from rag.claim_support import validate_screen,screen_claims,ClaimScreenUnavailable,segments


class ClaimSupportTests(unittest.TestCase):
    answer='## Synthetic draft\n\nFixture statement [source-1].\n\nThe synthetic exam is missing.'
    evidence=[{'chunk_id':'source-1','text':'The exact synthetic fixture statement is supported only for this test.','population':'adult','title':'Not real clinical evidence','source_date':'2024-01-01'}]

    def response(self,verdict='supported',quote=None):
        return {'checks':[
            {'segment_id':'S1','classification':'heading_or_nonclinical','verdict':'not_external_claim','supports':[],'reason_code':'not_external_claim'},
            {'segment_id':'S2','classification':'medical_claim','verdict':verdict,'supports':[{'chunk_id':'source-1','quote':quote or 'The exact synthetic fixture statement'}] if verdict=='supported' else [],'reason_code':'direct_support' if verdict=='supported' else 'missing_support'},
            {'segment_id':'S3','classification':'case_or_uncertainty','verdict':'not_external_claim','supports':[],'reason_code':'missing_case_data'},
        ]}

    def test_pass_requires_complete_segment_coverage_and_exact_quotes(self):
        result=validate_screen(self.answer,self.evidence,'adult',self.response())
        self.assertEqual(result['status'],'passed');self.assertTrue(result['screening_only']);self.assertFalse(result['clinical_validation']);self.assertTrue(result['clinician_approval_required'])
        self.assertEqual(result['counts']['supported'],1)
        for bad in ({'checks':self.response()['checks'][:-1]},{'checks':self.response()['checks']+[self.response()['checks'][0]]},self.response(quote='Invented supporting quotation')):
            with self.assertRaises(ValueError):validate_screen(self.answer,self.evidence,'adult',bad)

    def test_missing_support_for_associated_reference_is_blocked_not_approved(self):
        response=self.response();response['checks'][1]['supports']=[]
        result=validate_screen(self.answer,self.evidence,'adult',response)
        self.assertEqual(result['status'],'blocked');self.assertEqual(result['checks'][1]['verdict'],'unsupported')
        self.assertEqual(result['checks'][1]['reason_code'],'missing_support')

    def test_physician_citation_padding_retains_original_verdict_and_exact_missing_reference(self):
        evidence=self.evidence+[dict(self.evidence[0],chunk_id='source-2',text='Unrelated storage instructions in a synthetic fixture.')]
        answer=self.answer.replace('[source-1]','[source-1] [source-2]')
        result=validate_screen(answer,evidence,'adult',self.response())
        check=result['checks'][1]
        self.assertEqual(result['status'],'blocked')
        self.assertEqual(check['provider_verdict'],'supported')
        self.assertEqual(check['verdict'],'unsupported')
        self.assertEqual(check['citations_without_support'],['source-2'])
        self.assertEqual(check['reason_code'],'missing_support')


    def test_citation_membership_and_quote_source_mismatch_fail_closed(self):
        with self.assertRaises(ValueError):validate_screen(self.answer.replace('source-1','fabricated-id'),self.evidence,'adult',self.response())
        bad=self.response();bad['checks'][1]['supports'][0]['chunk_id']='another-source'
        with self.assertRaises(ValueError):validate_screen(self.answer,self.evidence,'adult',bad)
        bad=self.response();bad['checks'][1]['verdict']='not_external_claim';bad['checks'][1]['classification']='case_or_uncertainty'
        with self.assertRaises(ValueError):validate_screen(self.answer,self.evidence,'adult',bad)

    def test_population_mismatch_and_unknown_age_block_without_changing_model_judgment(self):
        result=validate_screen(self.answer,[dict(self.evidence[0],population='pediatric')],'adult',self.response())
        self.assertEqual(result['status'],'blocked');self.assertEqual(result['checks'][1]['reason_code'],'population_mismatch')
        result=validate_screen(self.answer,self.evidence,'unknown',self.response());self.assertEqual(result['status'],'blocked');self.assertEqual(result['counts']['uncertain'],1)
        result=validate_screen(self.answer,[dict(self.evidence[0],population='unknown')],'adult',self.response());self.assertEqual(result['status'],'blocked')

    def test_unsupported_or_uncertain_assertions_never_pass(self):
        for verdict in ('unsupported','uncertain'):
            result=validate_screen(self.answer,self.evidence,'adult',self.response(verdict))
            self.assertEqual(result['status'],'blocked');self.assertEqual(result['counts'][verdict],1)

    def test_one_bounded_real_adapter_call_uses_untrusted_data_boundary_and_records_usage(self):
        response=self.response();response['checks'][1]['supports']=[{'quote_ref':'E1:Q0'}]
        fixture=NS(choices=[NS(message=NS(content=json.dumps(response)),finish_reason='stop')],model='approved-fixture-snapshot',usage=NS(prompt_tokens=100,completion_tokens=50,total_tokens=150))
        with patch.dict(os.environ,{'LLM_BASE_URL':'https://provider.invalid/v1','LLM_API_KEY':'SYNTHETIC-KEY','LLM_MODEL':'approved-fixture'}),patch('rag.claim_support.OpenAI') as sdk:
            create=sdk.return_value.__enter__.return_value.chat.completions.create;create.return_value=fixture
            result=screen_claims(self.answer,self.evidence,'adult')
            create.assert_called_once();self.assertEqual(sdk.call_args.kwargs['timeout'],20);self.assertEqual(sdk.call_args.kwargs['max_retries'],0)
            self.assertEqual(create.call_args.kwargs['model'],'approved-fixture');self.assertIn('untrusted data',create.call_args.kwargs['messages'][0]['content']);self.assertTrue(create.call_args.kwargs['response_format']['json_schema']['strict'])
            self.assertEqual(result['usage']['total_tokens'],150);self.assertEqual(result['model_requested'],'approved-fixture')

    def test_unknown_registered_quote_or_wrong_source_fails_closed(self):
        for reference in ('E9:Q0','E1:invented'):
            response=self.response();response['checks'][1]['supports']=[{'quote_ref':reference}]
            fixture=NS(choices=[NS(message=NS(content=json.dumps(response)),finish_reason='stop')])
            with patch.dict(os.environ,{'LLM_BASE_URL':'https://provider.invalid/v1','LLM_API_KEY':'SYNTHETIC-KEY','LLM_MODEL':'fixture'}),patch('rag.claim_support.OpenAI') as sdk:
                sdk.return_value.__enter__.return_value.chat.completions.create.return_value=fixture
                with self.assertRaises(ClaimScreenUnavailable):screen_claims(self.answer,self.evidence,'adult')

    def test_provider_truncation_invalid_json_and_error_payloads_are_redacted(self):
        for fixture in (NS(choices=[NS(message=NS(content='SECRET-REFLECTION'),finish_reason='length')]),NS(choices=[NS(message=NS(content='SECRET-REFLECTION'),finish_reason='stop')]),RuntimeError('SECRET-REFLECTION')):
            with patch.dict(os.environ,{'LLM_BASE_URL':'https://provider.invalid/v1','LLM_API_KEY':'SYNTHETIC-KEY','LLM_MODEL':'approved-fixture'}),patch('rag.claim_support.OpenAI') as sdk:
                create=sdk.return_value.__enter__.return_value.chat.completions.create
                if isinstance(fixture,Exception):create.side_effect=fixture
                else:create.return_value=fixture
                with self.assertRaises(ClaimScreenUnavailable) as error:screen_claims(self.answer,self.evidence,'adult')
                self.assertNotIn('SECRET-REFLECTION',str(error.exception));self.assertEqual(create.call_count,1)

    def test_invalid_input_is_rejected_before_provider_transmission(self):
        with patch('rag.claim_support.OpenAI') as sdk:
            with self.assertRaises(ValueError):screen_claims('Unknown [made-up]',self.evidence,'adult')
            with self.assertRaises(ValueError):screen_claims(self.answer,self.evidence,'anything')
            sdk.assert_not_called()
        with self.assertRaises(ValueError):segments('a'*6001)
