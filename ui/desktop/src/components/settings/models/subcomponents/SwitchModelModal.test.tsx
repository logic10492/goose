import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { getAcpClient } from '../../../../acp/acpConnection';
import { acpGetProviderDetails } from '../../../../acp/providers';
import { IntlTestWrapper } from '../../../../i18n/test-utils';
import { fetchModelsForProviders } from '../modelInterface';
import { getPredefinedModelsFromEnv, shouldShowPredefinedModels } from '../predefinedModelsUtils';
import { SwitchModelModal } from './SwitchModelModal';

const changeModel = vi.hoisted(() => vi.fn());

vi.mock('../../../../acp/acpConnection', () => ({ getAcpClient: vi.fn() }));
vi.mock('../../../ModelAndProviderContext', () => ({
  useModelAndProvider: () => ({
    currentModel: 'gpt-6-astra',
    currentProvider: 'custom_dahetao',
    changeModel,
  }),
}));
vi.mock('../predefinedModelsUtils', () => ({
  shouldShowPredefinedModels: vi.fn(() => false),
  getPredefinedModelsFromEnv: vi.fn(() => []),
}));
vi.mock('../../../../utils/analytics', () => ({ trackModelChanged: vi.fn() }));

function clientFor(efforts?: string[], reasoning = true, preferred = 'max') {
  const entry = {
    providerId: 'custom_dahetao',
    providerName: 'Dahetao',
    description: 'Custom provider',
    defaultModel: 'gpt-6-astra',
    configured: true,
    available: true,
    providerType: 'Custom',
    category: 'model',
    acp: false,
    visibleInSetup: true,
    deprecated: false,
    configKeys: [],
    setupSteps: [],
    supportsRefresh: false,
    refreshing: false,
    models: [{ id: 'gpt-6-astra', name: 'GPT Astra', reasoning, thinkingEfforts: efforts }],
  };
  const client = {
    goose: {
      providersList_unstable: vi.fn().mockResolvedValue({ entries: [entry] }),
      preferencesRead_unstable: vi.fn().mockResolvedValue({
        values: [{ key: 'gooseThinkingEffort', value: preferred }],
      }),
      preferencesSave_unstable: vi.fn().mockResolvedValue({}),
    },
  };
  vi.mocked(getAcpClient).mockResolvedValue(
    client as unknown as Awaited<ReturnType<typeof getAcpClient>>
  );
  return client;
}

function renderModal() {
  render(<SwitchModelModal sessionId="session-1" onClose={vi.fn()} setView={vi.fn()} />, {
    wrapper: IntlTestWrapper,
  });
}

describe('SwitchModelModal thinking efforts', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    changeModel.mockResolvedValue(false);
    vi.mocked(shouldShowPredefinedModels).mockReturnValue(false);
    vi.mocked(getPredefinedModelsFromEnv).mockReturnValue([]);
  });

  it.each([
    [
      ['max', 'medium', 'high', 'low', 'off'],
      ['Off', 'Low', 'Medium', 'High', 'Max'],
    ],
    [
      ['high', 'max'],
      ['High', 'Max'],
    ],
    [undefined, ['Off', 'Low', 'Medium', 'High', 'Max']],
  ])('shows the available strengths for %j', async (efforts, labels) => {
    clientFor(efforts);
    const user = userEvent.setup();
    renderModal();
    await waitFor(() =>
      expect(screen.getByText('Max - No constraints on thinking depth')).toBeInTheDocument()
    );
    await user.click(screen.getByRole('combobox', { name: 'Thinking Effort' }));
    expect(
      screen.getAllByRole('option').map((option) => option.textContent?.split(' - ')[0])
    ).toEqual(labels);
    await user.click(screen.getByRole('option', { name: 'High - Deep reasoning (default)' }));
    await user.click(screen.getByRole('button', { name: 'Select model' }));
    await waitFor(() =>
      expect(changeModel).toHaveBeenCalledWith(
        'session-1',
        expect.objectContaining({
          name: 'gpt-6-astra',
          reasoning: true,
          request_params: { thinking_effort: 'high' },
        })
      )
    );
  });

  it('uses inventory capabilities in predefined mode and on model changes', async () => {
    clientFor(['high', 'max']);
    vi.mocked(shouldShowPredefinedModels).mockReturnValue(true);
    vi.mocked(getPredefinedModelsFromEnv).mockReturnValue([
      { name: 'gpt-6-astra', provider: 'custom_dahetao' },
      {
        name: 'other-model',
        provider: 'custom_dahetao',
        reasoning: true,
        thinking_efforts: ['low'],
      },
    ]);
    const user = userEvent.setup();
    renderModal();
    await waitFor(() =>
      expect(screen.getByText('Max - No constraints on thinking depth')).toBeInTheDocument()
    );
    await user.click(screen.getByText('other-model'));
    await waitFor(() =>
      expect(screen.getByText('Low - Minimal thinking, fastest responses')).toBeInTheDocument()
    );
    await user.click(screen.getByRole('button', { name: 'Select model' }));
    await waitFor(() =>
      expect(changeModel).toHaveBeenCalledWith(
        'session-1',
        expect.objectContaining({
          name: 'other-model',
          request_params: { thinking_effort: 'low' },
        })
      )
    );
  });

  it('submits the displayed supported fallback rather than an unavailable saved strength', async () => {
    const client = clientFor(['high'], true, 'low');
    const user = userEvent.setup();
    renderModal();
    await waitFor(() =>
      expect(screen.getByText('High - Deep reasoning (default)')).toBeInTheDocument()
    );
    await user.click(screen.getByRole('button', { name: 'Select model' }));
    await waitFor(() =>
      expect(changeModel).toHaveBeenCalledWith(
        'session-1',
        expect.objectContaining({
          request_params: { thinking_effort: 'high' },
        })
      )
    );
    expect(client.goose.preferencesSave_unstable).toHaveBeenCalledWith({
      values: [{ key: 'gooseThinkingEffort', value: 'high' }],
    });
  });

  it.each([
    [[], true],
    [['high'], false],
  ] as const)(
    'does not send effort when controls are unavailable (%j, %j)',
    async (efforts, reasoning) => {
      clientFor([...efforts], reasoning);
      const user = userEvent.setup();
      renderModal();
      await screen.findByText('gpt-6-astra');
      expect(screen.queryByRole('combobox', { name: 'Thinking Effort' })).not.toBeInTheDocument();
      await user.click(screen.getByRole('button', { name: 'Select model' }));
      await waitFor(() => expect(changeModel).toHaveBeenCalledOnce());
      expect(changeModel.mock.calls[0][1].request_params?.thinking_effort).toBeUndefined();
    }
  );

  it('preserves strengths when falling back to configured models after inventory failure', async () => {
    const client = clientFor(['max', 'high', 'off']);
    const details = await acpGetProviderDetails('custom_dahetao');
    client.goose.providersList_unstable.mockRejectedValue(new Error('inventory unavailable'));
    const [result] = await fetchModelsForProviders([details]);
    expect(result.models).toEqual([
      expect.objectContaining({
        name: 'gpt-6-astra',
        reasoning: true,
        thinking_efforts: ['max', 'high', 'off'],
      }),
    ]);
    expect(result.warning).not.toBeNull();
  });
});
