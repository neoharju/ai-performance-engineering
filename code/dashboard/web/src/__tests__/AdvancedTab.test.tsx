import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import { AdvancedTab } from '@/components/tabs/AdvancedTab';

const mockGetBenchRootConfig = jest.fn();
const mockSetBenchRootConfig = jest.fn();

jest.mock('@/lib/api', () => ({
  getCostCalculator: jest.fn(() => Promise.resolve(null)),
  getCostROI: jest.fn(() => Promise.resolve(null)),
  getAnalysisScaling: jest.fn(() => Promise.resolve(null)),
  getAnalysisPower: jest.fn(() => Promise.resolve(null)),
  getAnalysisCost: jest.fn(() => Promise.resolve(null)),
  simulateWhatIf: jest.fn(() => Promise.resolve(null)),
  getBenchRootConfig: (...args: unknown[]) => mockGetBenchRootConfig(...(args as any)),
  setBenchRootConfig: (...args: unknown[]) => mockSetBenchRootConfig(...(args as any)),
}));

jest.mock('@/components/Toast', () => ({
  useToast: () => ({ showToast: jest.fn() }),
}));

// LocalStorage mock
beforeAll(() => {
  const store: Record<string, string> = {};
  const localStorageMock = {
    getItem: (key: string) => (key in store ? store[key] : null),
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      Object.keys(store).forEach((k) => delete store[k]);
    },
  };
  Object.defineProperty(window, 'localStorage', {
    value: localStorageMock,
  });
});

describe('AdvancedTab - Project Root', () => {
  beforeEach(() => {
    mockGetBenchRootConfig.mockResolvedValue({
      success: true,
      bench_root: '/old/root',
      data_file: '/old/root/benchmark_test_results.json',
      benchmarks: 3,
      availability: {},
    });
    mockSetBenchRootConfig.mockResolvedValue({
      success: true,
      bench_root: '/new/root',
      data_file: '/new/root/benchmark_test_results.json',
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders project root card', async () => {
    render(<AdvancedTab />);
    await waitFor(() => expect(mockGetBenchRootConfig).toHaveBeenCalled());
    expect(screen.getAllByText(/Project Root/i).length).toBeGreaterThan(0);
    expect(screen.getByPlaceholderText(/path\/to\/project/i)).toBeInTheDocument();
  });

  it('applies a new project root', async () => {
    render(<AdvancedTab />);
    await waitFor(() => expect(mockGetBenchRootConfig).toHaveBeenCalled());

    const rootInput = screen.getByPlaceholderText(/path\/to\/project/i);
    const dataInput = screen.getByPlaceholderText(/benchmark_test_results\.json/i);
    await userEvent.clear(rootInput);
    await userEvent.type(rootInput, '/new/root');
    await userEvent.clear(dataInput);
    await userEvent.type(dataInput, '/new/root/results.json');

    const applyButton = screen.getByRole('button', { name: /Apply project root/i });
    await userEvent.click(applyButton);

    await waitFor(() =>
      expect(mockSetBenchRootConfig).toHaveBeenCalledWith({
        bench_root: '/new/root',
        data_file: '/new/root/results.json',
      }),
    );
  });
});
